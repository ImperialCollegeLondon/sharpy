"""Assembly of linearised UVLM system

S. Maraniello, 25 May 2018

Includes:
    - Boundary conditions methods:
        - AICs: allocate aero influence coefficient matrices of multi-surfaces
          configurations
        - ``nc_dqcdzeta_Sin_to_Sout``: derivative matrix of ``nc*dQ/dzeta``
          where Q is the induced velocity at the bound collocation points of one
          surface to another.
        - ``nc_dqcdzeta_coll``: assembles ``nc_dqcdzeta_coll_Sin_to_Sout`` matrices in
          multi-surfaces configurations
        - ``uc_dncdzeta``: assemble derivative matrix dnc/dzeta*Uc at bound collocation
          points
"""

import numpy as np
import scipy.sparse as sparse
import itertools

from sharpy.aero.utils.uvlmlib import dvinddzeta_cpp, eval_panel_cpp
import sharpy.linear.src.libsparse as libsp
import sharpy.linear.src.lib_dbiot as dbiot
import sharpy.linear.src.lib_ucdncdzeta as lib_ucdncdzeta
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout

# local indiced panel/vertices as per self.maps
dmver = [0, 1, 1, 0]  # delta to go from (m,n) panel to (m,n) vertices
dnver = [0, 0, 1, 1]
svec = [0, 1, 2, 3]  # seg. no.
avec = [0, 1, 2, 3]  # 1st vertex no.
bvec = [1, 2, 3, 0]  # 2nd vertex no.


def AICs(Surfs, Surfs_star, target='collocation', Project=True):
    """
    Given a list of bound (Surfs) and wake (Surfs_star) instances of
    surface.AeroGridSurface, returns the list of AIC matrices in the format:
        - AIC_list[ii][jj] contains the AIC from the bound surface Surfs[jj] to
        Surfs[ii].
        - AIC_star_list[ii][jj] contains the AIC from the wake surface Surfs[jj]
        to Surfs[ii].
    """

    AIC_list = []
    AIC_star_list = []

    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    for ss_out in range(n_surf):
        AIC_list_here = []
        AIC_star_list_here = []
        Surf_out = Surfs[ss_out]

        for ss_in in range(n_surf):
            # Bound surface
            Surf_in = Surfs[ss_in]
            AIC_list_here.append(Surf_in.get_aic_over_surface(
                Surf_out, target=target, Project=Project))
            # Wakes
            Surf_in = Surfs_star[ss_in]
            AIC_star_list_here.append(Surf_in.get_aic_over_surface(
                Surf_out, target=target, Project=Project))
        AIC_list.append(AIC_list_here)
        AIC_star_list.append(AIC_star_list_here)

    return AIC_list, AIC_star_list


def nc_dqcdzeta_Sin_to_Sout(Surf_in, Surf_out, Der_coll, Der_vert, Surf_in_bound):
    """
    Computes derivative matrix of
        nc*dQ/dzeta
    where Q is the induced velocity induced by bound surface Surf_in onto
    bound surface Surf_out. The panel normals of Surf_out are constant.

    The input/output are:
    - Der_coll of size (Kout,3*Kzeta_out): derivative due to the movement of
    collocation point on Surf_out.
    - Der_vert of size:
        - (Kout,3*Kzeta_in) if Surf_in_bound is True
        - (Kout,3*Kzeta_bound_in) if Surf_in_bound is False; Kzeta_bound_in is
        the number of vertices in the bound surface of whom Surf_out is the wake.

    Note that:
    - if Surf_in_bound is False, only the TE movement contributes to Der_vert.
    - if Surf_in_bound is False, the allocation of Der_coll could be speed-up by
    scanning only the wake segments along the chordwise direction, as on the
    others the net circulation is null.
    """

    # calc collocation points (and weights)
    if not hasattr(Surf_out, 'zetac'):
        Surf_out.generate_collocations()
    ZetaColl = Surf_out.zetac
    wcv_out = Surf_out.get_panel_wcv()

    # extract sizes / check matrices
    K_out = Surf_out.maps.K
    Kzeta_out = Surf_out.maps.Kzeta
    shape_zeta_out = Surf_out.maps.shape_vert_vect  # (3,M_out,N_out)
    K_in = Surf_in.maps.K
    Kzeta_in = Surf_in.maps.Kzeta

    assert Der_coll.shape == (K_out, 3 * Kzeta_out), 'Unexpected Der_coll shape'
    if Surf_in_bound:
        assert Der_vert.shape == (K_out, 3 * Kzeta_in), 'Unexpected Der_vert shape'
    else:
        # determine size of bound surface of which Surf_in is the wake
        Kzeta_bound_in = Der_vert.shape[1] // 3
        N_in = Surf_in.maps.N
        M_bound_in = Kzeta_bound_in // (N_in + 1) - 1

    # create mapping panels to vertices to loop
    Surf_out.maps.map_panels_to_vertices_1D_scalar()
    # Surf_in.maps.map_panels_to_vertices_1D_scalar()

    ##### loop collocation points
    for cc_out in range(K_out):

        # get (m,n) indices of collocation point
        mm_out = Surf_out.maps.ind_2d_pan_scal[0][cc_out]
        nn_out = Surf_out.maps.ind_2d_pan_scal[1][cc_out]

        # get coords and normal
        zetac_here = ZetaColl[:, mm_out, nn_out]  # .copy() # non-contiguous array !
        nc_here = Surf_out.normals[:, mm_out, nn_out]

        # get derivative of induced velocity w.r.t. zetac
        if Surf_in_bound:
            dvind_coll, dvind_vert = dvinddzeta_cpp(zetac_here, Surf_in,
                                                    is_bound=Surf_in_bound,
                                                    vortex_radius=Surf_in.vortex_radius)
        else:
            dvind_coll, dvind_vert = dvinddzeta_cpp(zetac_here, Surf_in,
                                                    is_bound=Surf_in_bound,
                                                    vortex_radius=Surf_in.vortex_radius,
                                                    M_in_bound=M_bound_in)

        ### Surf_in vertices contribution
        Der_vert[cc_out, :] += np.dot(nc_here, dvind_vert)

        ### Surf_out collocation point contribution
        # project
        dvindnorm_coll = np.dot(nc_here, dvind_coll)

        # loop panel vertices
        for vv, dm, dn in zip(range(4), dmver, dnver):
            mm_v, nn_v = mm_out + dm, nn_out + dn
            ii_v = [np.ravel_multi_index(
                (cc, mm_v, nn_v), shape_zeta_out) for cc in range(3)]
            Der_coll[cc_out, ii_v] += wcv_out[vv] * dvindnorm_coll

    return Der_coll, Der_vert


def nc_dqcdzeta(Surfs, Surfs_star, Merge=False):
    r"""
    Produces a list of derivative matrix

    .. math:: \frac{\partial(\mathcal{A}\boldsymbol{\Gamma}_0)}{\partial\boldsymbol{\zeta}}

    where :math:`\mathcal{A}` is the aerodynamic influence coefficient matrix at the bound
    surfaces collocation point, assuming constant panel norm.

    Each list is such that:

        - the ``ii``-th element is associated to the ``ii``-th bound surface collocation
          point, and will contain a sub-list such that:
            - the ``j``-th element of the sub-list is the ``dAIC_dzeta`` matrices w.r.t. the
              ``zeta`` d.o.f. of the ``j``-th bound surface.

    Hence, ``DAIC*[ii][jj]`` will have size ``K_ii x Kzeta_jj``

    If ``Merge`` is ``True``, the derivatives due to collocation points movement are added
    to ``Dvert`` to minimise storage space.

    To do:

        - Dcoll is highly sparse, exploit?
    """

    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    DAICcoll = []
    DAICvert = []

    ### loop output (bound) surfaces
    for ss_out in range(n_surf):

        # define output bound surface size
        Surf_out = Surfs[ss_out]
        K_out = Surf_out.maps.K
        Kzeta_out = Surf_out.maps.Kzeta

        # derivatives w.r.t collocation points: all the in surface scanned will
        # manipulate this matrix, as the collocation points are on Surf_out
        Dcoll = np.zeros((K_out, 3 * Kzeta_out))
        # derivatives w.r.t. panel coordinates will affect dof on bound Surf_in
        # (not wakes)
        DAICvert_sub = []

        # loop input surfaces:
        for ss_in in range(n_surf):
            ##### bound
            Surf_in = Surfs[ss_in]
            Kzeta_in = Surf_in.maps.Kzeta

            # compute terms
            Dvert = np.zeros((K_out, 3 * Kzeta_in))
            Dcoll, Dvert = nc_dqcdzeta_Sin_to_Sout(
                Surf_in, Surf_out, Dcoll, Dvert, Surf_in_bound=True)

            ##### wake:
            Surf_in = Surfs_star[ss_in]
            Dcoll, Dvert = nc_dqcdzeta_Sin_to_Sout(
                Surf_in, Surf_out, Dcoll, Dvert, Surf_in_bound=False)
            DAICvert_sub.append(Dvert)

        if Merge:
            DAICvert_sub[ss_out] += Dcoll
            DAICvert.append(DAICvert_sub)
        else:
            DAICcoll.append(Dcoll)
            DAICvert.append(DAICvert_sub)

    if Merge:
        return DAICvert
    else:
        return DAICcoll, DAICvert


# end

def nc_domegazetadzeta(Surfs, Surfs_star):
    """
    Produces a list of derivative matrix d(omaga x zeta)/dzeta, where omega is
    the rotation speed of the A FoR,
    ASSUMING constant panel norm.

    Each list is such that:
    - the ii-th element is associated to the ii-th bound surface collocation
    point, and will contain a sub-list such that:
        - the j-th element of the sub-list is the dAIC_dzeta matrices w.r.t. the
        zeta d.o.f. of the j-th bound surface.
    Hence, DAIC*[ii][jj] will have size K_ii x Kzeta_jj

    call: ncDOmegaZetavert = nc_domegazetadzeta(Surfs,Surfs_star)
    """
    n_surf = len(Surfs)

    ncDOmegaZetacoll = []
    ncDOmegaZetavert = []

    ### loop output (bound) surfaces
    for ss in range(n_surf):

        # define output bound surface size
        Surf = Surfs[ss]
        skew_omega = algebra.skew(Surf.omega)
        K = Surf.maps.K  # K_out = M*N (number of panels)
        Kzeta = Surf.maps.Kzeta  # Kzeta_out = (M+1)*(N+1) (number of vertices/edges)
        wcv = Surf.get_panel_wcv()
        shape_zeta = Surf.maps.shape_vert_vect  # (3,M,N)

        # The derivatives only depend on the studied surface (Surf)
        ncDvert = np.zeros((K, 3 * Kzeta))

        ##### loop collocation points
        for cc in range(K):

            # get (m,n) indices of collocation point
            mm = Surf.maps.ind_2d_pan_scal[0][cc]
            nn = Surf.maps.ind_2d_pan_scal[1][cc]

            # get normal
            nc_here = Surf.normals[:, mm, nn]

            nc_skew_omega = -1. * np.dot(nc_here, skew_omega)

            # loop panel vertices
            for vv, dm, dn in zip(range(4), dmver, dnver):
                mm_v, nn_v = mm + dm, nn + dn
                ii_v = [np.ravel_multi_index(
                    (comp, mm_v, nn_v), shape_zeta) for comp in range(3)]

                ncDvert[cc, ii_v] += nc_skew_omega

        ncDOmegaZetavert.append(ncDvert)

    return ncDOmegaZetavert


def uc_dncdzeta(Surf):
    r"""
    Build derivative of

    ..  math:: \boldsymbol{u}_c\frac{\partial\boldsymbol{n}_c}{\partial\boldsymbol{zeta}}

    where :math:`\boldsymbol{u}_c` is the total velocity at the
    collocation points.

    Args:
        Surf (surface.AerogridSurface): the input can also be a list of :class:`surface.AerogridSurface`

    References:
        - :module:`linear.develop_sym.linsum_Wnc`
        - :module:`lib_ucdncdzeta`
    """

    if type(Surf) is list:
        n_surf = len(Surf)
        DerList = []
        for ss in range(n_surf):
            DerList.append(uc_dncdzeta(Surf[ss]))
        return DerList
    else:
        if (not hasattr(Surf, 'u_ind_coll')) or (Surf.u_input_coll is None):
            raise NameError(
                'Surf does not have the required attributes\nu_ind_coll\nu_input_coll')

    Map = Surf.maps
    K, Kzeta = Map.K, Map.Kzeta
    Der = np.zeros((K, 3 * Kzeta))

    # map panel to vertice
    if not hasattr(Map.Mpv, 'Mpv1d_scalar'):
        Map.map_panels_to_vertices_1D_scalar()
    if not hasattr(Map.Mpv, 'Mpv'):
        Map.map_panels_to_vertices()

    # map u_normal 2d to 1d
    # map_panels_1d_to_2d=np.unravel_index(range(K),
    # 						   				  dims=Map.shape_pan_scal,order='C')
    # for ii in range(K):

    for ii in Map.ind_1d_pan_scal:

        # panel m,n coordinates
        m_pan, n_pan = Map.ind_2d_pan_scal[0][ii], Map.ind_2d_pan_scal[1][ii]
        # extract u_input_coll
        u_tot_coll_here = \
            Surf.u_input_coll[:, m_pan, n_pan] + Surf.u_ind_coll[:, m_pan, n_pan]

        # find vertices
        mpv = Map.Mpv[m_pan, n_pan, :, :]

        # extract m,n coordinates of vertices
        zeta00 = Surf.zeta[:, mpv[0, 0], mpv[0, 1]]
        zeta01 = Surf.zeta[:, mpv[1, 0], mpv[1, 1]]
        zeta02 = Surf.zeta[:, mpv[2, 0], mpv[2, 1]]
        zeta03 = Surf.zeta[:, mpv[3, 0], mpv[3, 1]]

        # calculate derivative
        Dlocal = lib_ucdncdzeta.eval(zeta00, zeta01, zeta02, zeta03, u_tot_coll_here)

        for vv in range(4):
            # find 1D position of vertices
            jj = Map.Mpv1d_scalar[ii, vv]

            # allocate derivatives
            Der[ii, jj] = Dlocal[vv, 0]  # w.r.t. x
            Der[ii, jj + Kzeta] = Dlocal[vv, 1]  # w.r.t. y
            Der[ii, jj + 2 * Kzeta] = Dlocal[vv, 2]  # w.r.t. z

    return Der


def dfqsdgamma_vrel0(Surfs, Surfs_star):
    """
    Assemble derivative of quasi-steady force w.r.t. gamma with fixed relative
    velocity - the changes in induced velocities due to gamma are not accounted
    for. The routine exploits the get_joukovski_qs method insude the
    AeroGridSurface class
    """

    Der_list = []
    Der_star_list = []

    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    for ss in range(n_surf):

        Surf = Surfs[ss]
        if not hasattr(Surf, 'u_ind_seg'):
            raise NameError('Induced velocities at segments missing')
        if Surf.u_input_seg is None:
            raise NameError('Input velocities at segments missing')
        if not hasattr(Surf, 'fqs_seg'):
            Surf.get_joukovski_qs(gammaw_TE=Surfs_star[ss].gamma[0, :])

        M, N = Surf.maps.M, Surf.maps.N
        K = Surf.maps.K
        Kzeta = Surf.maps.Kzeta
        shape_fqs = Surf.maps.shape_vert_vect  # (3,M+1,N+1)

        ##### unit gamma contribution of BOUND panels
        Der = np.zeros((3 * Kzeta, K))

        # loop panels (input, i.e. matrix columns)
        for pp_in in range(K):
            # get (m,n) indices of panel
            mm_in = Surf.maps.ind_2d_pan_scal[0][pp_in]
            nn_in = Surf.maps.ind_2d_pan_scal[1][pp_in]

            # zetav_here=Surf.get_panel_vertices_coords(mm_in,nn_in)
            for ll, aa, bb in zip(svec, avec, bvec):
                # import libuvlm
                # dfhere=libuvlm.joukovski_qs_segment(
                # 	zetaA=zetav_here[aa,:],zetaB=zetav_here[bb,:],
                # 	v_mid=Surf.u_ind_seg[:,ll,mm_in,nn_in]+\
                # 		  Surf.u_input_seg[:,ll,mm_in,nn_in],
                # 	gamma=1.0,fact=0.5*Surf.rho)
                df = 0.5 * Surf.fqs_seg_unit[:, ll, mm_in, nn_in]
                # assert np.abs(np.max(dfhere-df))<1e-13,'something is wrong'

                # get vertices m,n indices
                mm_a, nn_a = mm_in + dmver[aa], nn_in + dnver[aa]
                mm_b, nn_b = mm_in + dmver[bb], nn_in + dnver[bb]

                # get vertices 1d index
                ii_a = [np.ravel_multi_index(
                    (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
                ii_b = [np.ravel_multi_index(
                    (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]
                Der[ii_a, pp_in] += df
                Der[ii_b, pp_in] += df

        Der_list.append(Der)

        ##### unit gamma contribution of WAKE TE segments
        # Note: the force due to the wake is attached to Surf when
        # get_joukovski_qs is acalled
        M_star, N_star = Surfs_star[ss].maps.M, Surfs_star[ss].maps.N
        K_star = Surfs_star[ss].maps.K
        shape_in = Surfs_star[ss].maps.shape_pan_scal  # (M_star,N_star)

        Der_star = np.zeros((3 * Kzeta, K_star))

        assert N == N_star, \
            'trying to associate wrong wake to current bound surface!'

        # loop bound panels
        for nn_in in range(N):
            pp_in = np.ravel_multi_index((0, nn_in), shape_in)

            df = 0.5 * Surf.fqs_wTE_unit[:, nn_in]

            # get TE bound vertices m,n indices
            mm_a, nn_a = M, nn_in + dnver[aa]
            mm_b, nn_b = M, nn_in + dnver[bb]

            # get vertices 1d index
            ii_a = [np.ravel_multi_index(
                (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
            ii_b = [np.ravel_multi_index(
                (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]
            Der_star[ii_a, pp_in] += df
            Der_star[ii_b, pp_in] += df

        Der_star_list.append(Der_star)

    return Der_list, Der_star_list


def dfqsdzeta_vrel0(Surfs, Surfs_star):
    """
    Assemble derivative of quasi-steady force w.r.t. zeta with fixed relative
    velocity - the changes in induced velocities due to zeta over the surface
    inducing the velocity are not accounted for. The routine exploits the
    available relative velocities at the mid-segment points
    """

    Der_list = []
    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    for ss in range(n_surf):

        Surf = Surfs[ss]
        if not hasattr(Surf, 'u_ind_seg'):
            raise NameError('Induced velocities at segments missing')
        if Surf.u_input_seg is None:
            raise NameError('Input velocities at segments missing')

        M, N = Surf.maps.M, Surf.maps.N
        K = Surf.maps.K
        Kzeta = Surf.maps.Kzeta
        shape_fqs = Surf.maps.shape_vert_vect  # (3,M+1,N+1)

        ##### unit gamma contribution of BOUND panels
        Der = np.zeros((3 * Kzeta, 3 * Kzeta))

        # loop panels (input, i.e. matrix columns)
        for pp_in in range(K):
            # get (m,n) indices of panel
            mm_in = Surf.maps.ind_2d_pan_scal[0][pp_in]
            nn_in = Surf.maps.ind_2d_pan_scal[1][pp_in]

            for ll, aa, bb in zip(svec, avec, bvec):
                vrel_seg = (Surf.u_input_seg[:, ll, mm_in, nn_in] +
                            Surf.u_ind_seg[:, ll, mm_in, nn_in])
                Df = algebra.skew((0.5 * Surf.rho * Surf.gamma[mm_in, nn_in]) * vrel_seg)

                # get vertices m,n indices
                mm_a, nn_a = mm_in + dmver[aa], nn_in + dnver[aa]
                mm_b, nn_b = mm_in + dmver[bb], nn_in + dnver[bb]

                # get vertices 1d index
                ii_a = [np.ravel_multi_index(
                    (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
                ii_b = [np.ravel_multi_index(
                    (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]
                Der[np.ix_(ii_a, ii_a)] += -Df
                Der[np.ix_(ii_b, ii_a)] += -Df
                Der[np.ix_(ii_a, ii_b)] += Df
                Der[np.ix_(ii_b, ii_b)] += Df

        ##### contribution of WAKE TE segments.
        # This is added to Der, as only the bound vertices are included in the
        # input

        # loop TE bound segment but:
        # - using wake gamma
        # - using orientation of wake panel
        for nn_in in range(N):
            # get velocity at seg.3 of wake TE
            vrel_seg = (Surf.u_input_seg[:, 1, M - 1, nn_in] + Surf.u_ind_seg[:, 1, M - 1, nn_in])
            Df = Df = algebra.skew(
                (0.5 * Surfs_star[ss].rho * Surfs_star[ss].gamma[0, nn_in]) * vrel_seg)

            # get TE bound vertices m,n indices
            nn_a = nn_in + dnver[2]
            nn_b = nn_in + dnver[1]
            # get vertices 1d index on bound
            ii_a = [np.ravel_multi_index(
                (cc, M, nn_a), shape_fqs) for cc in range(3)]
            ii_b = [np.ravel_multi_index(
                (cc, M, nn_b), shape_fqs) for cc in range(3)]

            Der[np.ix_(ii_a, ii_a)] += -Df
            Der[np.ix_(ii_b, ii_a)] += -Df
            Der[np.ix_(ii_a, ii_b)] += Df
            Der[np.ix_(ii_b, ii_b)] += Df
        Der_list.append(Der)

    return Der_list


def dfqsduinput(Surfs, Surfs_star):
    """
    Assemble derivative of quasi-steady force w.r.t. external input velocity.
    """

    Der_list = []
    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    for ss in range(n_surf):

        Surf = Surfs[ss]
        if Surf.u_input_seg is None:
            raise NameError('Input velocities at segments missing')

        M, N = Surf.maps.M, Surf.maps.N
        K = Surf.maps.K
        Kzeta = Surf.maps.Kzeta
        shape_fqs = Surf.maps.shape_vert_vect  # (3,M+1,N+1)

        ##### unit gamma contribution of BOUND panels
        Der = np.zeros((3 * Kzeta, 3 * Kzeta))

        # loop panels (input, i.e. matrix columns)
        for pp_in in range(K):
            # get (m,n) indices of panel
            mm_in = Surf.maps.ind_2d_pan_scal[0][pp_in]
            nn_in = Surf.maps.ind_2d_pan_scal[1][pp_in]

            # get panel vertices
            # zetav_here=Surf.get_panel_vertices_coords(mm_in,nn_in)
            zetav_here = Surf.zeta[:, [mm_in + 0, mm_in + 1, mm_in + 1, mm_in + 0],
                         [nn_in + 0, nn_in + 0, nn_in + 1, nn_in + 1]].T

            for ll, aa, bb in zip(svec, avec, bvec):
                # get segment
                lv = zetav_here[bb, :] - zetav_here[aa, :]
                Df = algebra.skew((-0.25 * Surf.rho * Surf.gamma[mm_in, nn_in]) * lv)

                # get vertices m,n indices
                mm_a, nn_a = mm_in + dmver[aa], nn_in + dnver[aa]
                mm_b, nn_b = mm_in + dmver[bb], nn_in + dnver[bb]

                # get vertices 1d index
                ii_a = [np.ravel_multi_index(
                    (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
                ii_b = [np.ravel_multi_index(
                    (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]
                Der[np.ix_(ii_a, ii_a)] += Df
                Der[np.ix_(ii_b, ii_a)] += Df
                Der[np.ix_(ii_a, ii_b)] += Df
                Der[np.ix_(ii_b, ii_b)] += Df

        ##### contribution of WAKE TE segments.
        # This is added to Der, as only velocities at the bound vertices are
        # included in the input of the state-space model

        # loop TE bound segment but:
        # - using wake gamma
        # - using orientation of wake panel
        for nn_in in range(N):
            # get TE bound vertices m,n indices
            nn_a = nn_in + dnver[2]
            nn_b = nn_in + dnver[1]

            # get segment
            lv = Surf.zeta[:, M, nn_b] - Surf.zeta[:, M, nn_a]
            Df = algebra.skew((-0.25 * Surf.rho * Surf.gamma[mm_in, nn_in]) * lv)

            # get vertices 1d index on bound
            ii_a = [np.ravel_multi_index(
                (cc, M, nn_a), shape_fqs) for cc in range(3)]
            ii_b = [np.ravel_multi_index(
                (cc, M, nn_b), shape_fqs) for cc in range(3)]

            Der[np.ix_(ii_a, ii_a)] += Df
            Der[np.ix_(ii_b, ii_a)] += Df
            Der[np.ix_(ii_a, ii_b)] += Df
            Der[np.ix_(ii_b, ii_b)] += Df
        Der_list.append(Der)

    return Der_list


def dfqsdzeta_omega(Surfs, Surfs_star):
    """
    Assemble derivative of quasi-steady force w.r.t. to zeta
    The contribution implemented is related with the omega x zeta term
    call: Der_list = dfqsdzeta_omega(Surfs,Surfs_star)
    """

    Der_list = []
    n_surf = len(Surfs)

    for ss in range(n_surf):

        Surf = Surfs[ss]
        skew_omega = algebra.skew(Surf.omega)
        M, N = Surf.maps.M, Surf.maps.N
        K = Surf.maps.K
        Kzeta = Surf.maps.Kzeta
        shape_fqs = Surf.maps.shape_vert_vect  # (3,M+1,N+1)

        ##### omega x zeta contribution
        Der = np.zeros((3 * Kzeta, 3 * Kzeta))

        # loop panels (input, i.e. matrix columns)
        for pp_in in range(K):
            # get (m,n) indices of panel
            mm_in = Surf.maps.ind_2d_pan_scal[0][pp_in]
            nn_in = Surf.maps.ind_2d_pan_scal[1][pp_in]

            # get panel vertices
            zetav_here = Surf.zeta[:, [mm_in + 0, mm_in + 1, mm_in + 1, mm_in + 0],
                         [nn_in + 0, nn_in + 0, nn_in + 1, nn_in + 1]].T

            for ll, aa, bb in zip(svec, avec, bvec):
                # get segment
                lv = zetav_here[bb, :] - zetav_here[aa, :]
                Df = (0.25 * Surf.rho * Surf.gamma[mm_in, nn_in]) * algebra.skew(lv).dot(skew_omega)

                # get vertices m,n indices
                mm_a, nn_a = mm_in + dmver[aa], nn_in + dnver[aa]
                mm_b, nn_b = mm_in + dmver[bb], nn_in + dnver[bb]

                # get vertices 1d index
                ii_a = [np.ravel_multi_index(
                    (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
                ii_b = [np.ravel_multi_index(
                    (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]
                Der[np.ix_(ii_a, ii_a)] += Df
                Der[np.ix_(ii_b, ii_a)] += Df
                Der[np.ix_(ii_a, ii_b)] += Df
                Der[np.ix_(ii_b, ii_b)] += Df

        ##### contribution of WAKE TE segments.
        # This is added to Der, as only velocities at the bound vertices are
        # included in the input of the state-space model

        # loop TE bound segment but:
        # - using wake gamma
        # - using orientation of wake panel
        for nn_in in range(N):
            # get TE bound vertices m,n indices
            nn_a = nn_in + dnver[2]
            nn_b = nn_in + dnver[1]

            # get segment
            lv = Surf.zeta[:, M, nn_b] - Surf.zeta[:, M, nn_a]
            Df = (0.25 * Surf.rho * Surf.gamma[mm_in, nn_in]) * algebra.skew(lv).dot(skew_omega)

            # get vertices 1d index on bound
            ii_a = [np.ravel_multi_index(
                (cc, M, nn_a), shape_fqs) for cc in range(3)]
            ii_b = [np.ravel_multi_index(
                (cc, M, nn_b), shape_fqs) for cc in range(3)]

            Der[np.ix_(ii_a, ii_a)] += Df
            Der[np.ix_(ii_b, ii_a)] += Df
            Der[np.ix_(ii_a, ii_b)] += Df
            Der[np.ix_(ii_b, ii_b)] += Df

        Der_list.append(Der)

    return Der_list


def dfqsdvind_gamma(Surfs, Surfs_star):
    """
    Assemble derivative of quasi-steady force w.r.t. induced velocities changes
    due to gamma.
    Note: the routine is memory consuming but avoids unnecessary computations.
    """

    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    ### compute all influence coeff matrices (high RAM, low CPU)
    # AIC_list,AIC_star_list=AICs(Surfs,Surfs_star,target='segments',Project=False)

    Der_list = []
    Der_star_list = []
    for ss_out in range(n_surf):

        Surf_out = Surfs[ss_out]
        M_out, N_out = Surf_out.maps.M, Surf_out.maps.N
        K_out = Surf_out.maps.K
        Kzeta_out = Surf_out.maps.Kzeta
        shape_fqs = Surf_out.maps.shape_vert_vect  # (3,M+1,N+1)

        # get AICs over Surf_out
        AICs = []
        AICs_star = []
        for ss_in in range(n_surf):
            AICs.append(Surfs[ss_in].get_aic_over_surface(
                Surf_out, target='segments', Project=False))
            AICs_star.append(Surfs_star[ss_in].get_aic_over_surface(
                Surf_out, target='segments', Project=False))

        # allocate all derivative matrices
        Der_list_sub = []
        Der_star_list_sub = []
        for ss_in in range(n_surf):
            # bound
            K_in = Surfs[ss_in].maps.K
            Der_list_sub.append(np.zeros((3 * Kzeta_out, K_in)))
            # wake
            K_in = Surfs_star[ss_in].maps.K
            Der_star_list_sub.append(np.zeros((3 * Kzeta_out, K_in)))

        ### loop bound panels
        for pp_out in range(K_out):
            # get (m,n) indices of panel
            mm_out = Surf_out.maps.ind_2d_pan_scal[0][pp_out]
            nn_out = Surf_out.maps.ind_2d_pan_scal[1][pp_out]
            # get panel vertices
            # zetav_here=Surf_out.get_panel_vertices_coords(mm_out,nn_out)
            zetav_here = Surf_out.zeta[:, [mm_out + 0, mm_out + 1, mm_out + 1, mm_out + 0],
                         [nn_out + 0, nn_out + 0, nn_out + 1, nn_out + 1]].T

            for ll, aa, bb in zip(svec, avec, bvec):

                # get segment
                lv = zetav_here[bb, :] - zetav_here[aa, :]
                Lskew = algebra.skew((-0.5 * Surf_out.rho * Surf_out.gamma[mm_out, nn_out]) * lv)

                # get vertices m,n indices
                mm_a, nn_a = mm_out + dmver[aa], nn_out + dnver[aa]
                mm_b, nn_b = mm_out + dmver[bb], nn_out + dnver[bb]

                # get vertices 1d index
                ii_a = [np.ravel_multi_index(
                    (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
                ii_b = [np.ravel_multi_index(
                    (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]

                # update all derivatives
                for ss_in in range(n_surf):
                    # derivatives: size (3,K_in)
                    Dfs = np.dot(Lskew, AICs[ss_in][:, :, ll, mm_out, nn_out])
                    Dfs_star = np.dot(Lskew, AICs_star[ss_in][:, :, ll, mm_out, nn_out])
                    # allocate
                    Der_list_sub[ss_in][ii_a, :] += Dfs
                    Der_list_sub[ss_in][ii_b, :] += Dfs
                    Der_star_list_sub[ss_in][ii_a, :] += Dfs_star
                    Der_star_list_sub[ss_in][ii_b, :] += Dfs_star

        ### loop again trailing edge
        # here we add the Gammaw_0*rho*skew(lv)*dvind/dgamma contribution hence:
        # - we use Gammaw_0 over the TE
        # - we run along the positive direction as defined in the first row of
        # wake panels
        for nn_out in range(N_out):

            # get TE bound vertices m,n indices
            nn_a = nn_out + dnver[2]
            nn_b = nn_out + dnver[1]

            # get segment
            lv = Surf_out.zeta[:, M_out, nn_b] - Surf_out.zeta[:, M_out, nn_a]
            Lskew = algebra.skew((-0.5 * Surf_out.rho * Surfs_star[ss_out].gamma[0, nn_out]) * lv)

            # get vertices 1d index on bound
            ii_a = [np.ravel_multi_index(
                (cc, M_out, nn_a), shape_fqs) for cc in range(3)]
            ii_b = [np.ravel_multi_index(
                (cc, M_out, nn_b), shape_fqs) for cc in range(3)]

            # update all derivatives
            for ss_in in range(n_surf):
                # derivatives: size (3,K_in)
                Dfs = np.dot(Lskew, AICs[ss_in][:, :, 1, M_out - 1, nn_out])
                Dfs_star = np.dot(Lskew, AICs_star[ss_in][:, :, 1, M_out - 1, nn_out])
                # allocate
                Der_list_sub[ss_in][ii_a, :] += Dfs
                Der_list_sub[ss_in][ii_b, :] += Dfs
                Der_star_list_sub[ss_in][ii_a, :] += Dfs_star
                Der_star_list_sub[ss_in][ii_b, :] += Dfs_star

        Der_list.append(Der_list_sub)
        Der_star_list.append(Der_star_list_sub)

    return Der_list, Der_star_list


def dvinddzeta(zetac, Surf_in, IsBound, M_in_bound=None):
    """
    Produces derivatives of induced velocity by Surf_in w.r.t. the zetac point.
    Derivatives are divided into those associated to the movement of zetac, and
    to the movement of the Surf_in vertices (DerVert).

    If Surf_in is bound (IsBound==True), the circulation over the TE due to the
    wake is not included in the input.

    If Surf_in is a wake (IsBound==False), derivatives w.r.t. collocation
    points are computed ad the TE contribution on DerVert. In this case, the
    chordwise paneling Min_bound of the associated input is required so as to
    calculate Kzeta and correctly allocate the derivative matrix.

    The output derivatives are:
    - Dercoll: 3 x 3 matrix
    - Dervert: 3 x 3*Kzeta (if Surf_in is a wake, Kzeta is that of the bound)

    Warning:
    zetac must be contiguously stored!
    """

    M_in, N_in = Surf_in.maps.M, Surf_in.maps.N
    Kzeta_in = Surf_in.maps.Kzeta
    shape_zeta_in = (3, M_in + 1, N_in + 1)

    # allocate matrices
    Dercoll = np.zeros((3, 3))

    if IsBound:
        """ Bound: scan everthing, and include every derivative. The TE is not
        scanned twice"""

        Dervert = np.zeros((3, 3 * Kzeta_in))

        for pp_in in itertools.product(range(0, M_in), range(0, N_in)):
            mm_in, nn_in = pp_in
            # zeta_panel_in=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
            zeta_panel_in = Surf_in.zeta[:, [mm_in + 0, mm_in + 1, mm_in + 1, mm_in + 0],
                            [nn_in + 0, nn_in + 0, nn_in + 1, nn_in + 1]].T
            # get local derivatives
            der_zetac, der_zeta_panel = eval_panel_cpp(
                zetac, zeta_panel_in, Surf_in.vortex_radius, gamma_pan=Surf_in.gamma[mm_in, nn_in])
            ### Mid-segment point contribution
            Dercoll += der_zetac
            ### Panel vertices contribution
            for vv_in in range(4):
                # get vertices m,n indices
                mm_v, nn_v = mm_in + dmver[vv_in], nn_in + dnver[vv_in]
                # get vertices 1d index
                jj_v = [np.ravel_multi_index(
                    (cc, mm_v, nn_v), shape_zeta_in) for cc in range(3)]
                Dervert[:, jj_v] += der_zeta_panel[vv_in, :, :]

    else:
        """
        All segments are scanned when computing the contrib. Dercoll. The
        TE is scanned a second time to include the contrib. due to the TE
        elements moviment. The Dervert shape is computed using the chordwse
        paneling of the associated bound surface (M_in_bound).
        """

        Kzeta_in_bound = (M_in_bound + 1) * (N_in + 1)
        Dervert = np.zeros((3, 3 * Kzeta_in_bound))

        ### loop all panels (coll. contrib)
        for pp_in in itertools.product(range(0, M_in), range(0, N_in)):
            mm_in, nn_in = pp_in
            # zeta_panel_in=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
            zeta_panel_in = Surf_in.zeta[:, [mm_in + 0, mm_in + 1, mm_in + 1, mm_in + 0],
                            [nn_in + 0, nn_in + 0, nn_in + 1, nn_in + 1]].T
            # get local derivatives
            der_zetac = dbiot.eval_panel_cpp_coll(
                zetac, zeta_panel_in, Surf_in.vortex_radius, gamma_pan=Surf_in.gamma[mm_in, nn_in])
            # der_zetac_fast=dbiot.eval_panel_fast_coll(
            # 		zetac,zeta_panel_in,gamma_pan=Surf_in.gamma[mm_in,nn_in])
            # if np.max(np.abs(der_zetac-der_zetac_fast))>1e-10:
            # 	embed()

            ### Mid-segment point contribution
            Dercoll += der_zetac

        ### Re-scan the TE to include vertex contrib.
        # vertex 0 of wake is vertex 1 of bound (local no.)
        # vertex 3 of wake is vertex 2 of bound (local no.)
        vvec = [0, 3]  # vertices to include
        dn = [0, 1]  # delta to go from (m,n) panel to (m,n) vertices (on bound)

        shape_zeta_in_bound = (3, M_in_bound + 1, N_in + 1)
        for nn_in in range(N_in):
            # zeta_panel_in=Surf_in.get_panel_vertices_coords(0,nn_in)
            zeta_panel_in = Surf_in.zeta[:, [0, 1, 1, 0],
                            [nn_in + 0, nn_in + 0, nn_in + 1, nn_in + 1]].T
            # get local derivatives
            _, der_zeta_panel = eval_panel_cpp(
                zetac, zeta_panel_in, Surf_in.vortex_radius, gamma_pan=Surf_in.gamma[0, nn_in])

            for vv in range(2):
                nn_v = nn_in + dn[vv]
                jj_v = []
                for cc in range(3):
                    jj_v.append(np.ravel_multi_index(
                        (cc, M_in_bound, nn_v), shape_zeta_in_bound))
                Dervert[:, jj_v] += der_zeta_panel[vvec[vv], :, :]

    return Dercoll, Dervert


def dfqsdvind_zeta(Surfs, Surfs_star):
    """
    Assemble derivative of quasi-steady force w.r.t. induced velocities changes
    due to zeta.
    """

    n_surf = len(Surfs)
    assert len(Surfs_star) == n_surf, \
        'Number of bound and wake surfaces much be equal'

    # allocate
    Dercoll_list = []
    Dervert_list = []
    for ss_out in range(n_surf):
        Kzeta_out = Surfs[ss_out].maps.Kzeta
        Dercoll_list.append(np.zeros((3 * Kzeta_out, 3 * Kzeta_out)))
        Dervert_list_sub = []
        for ss_in in range(n_surf):
            Kzeta_in = Surfs[ss_in].maps.Kzeta
            Dervert_list_sub.append(np.zeros((3 * Kzeta_out, 3 * Kzeta_in)))
        Dervert_list.append(Dervert_list_sub)

    for ss_out in range(n_surf):

        Surf_out = Surfs[ss_out]
        M_out, N_out = Surf_out.maps.M, Surf_out.maps.N
        K_out = Surf_out.maps.K
        Kzeta_out = Surf_out.maps.Kzeta
        shape_fqs = Surf_out.maps.shape_vert_vect  # (3,M+1,N+1)
        Dercoll = Dercoll_list[ss_out]  # <--link

        ### Loop out (bound) surface panels
        for pp_out in itertools.product(range(0, M_out), range(0, N_out)):
            mm_out, nn_out = pp_out
            # zeta_panel_out=Surf_out.get_panel_vertices_coords(mm_out,nn_out)
            zeta_panel_out = Surf_out.zeta[:, [mm_out + 0, mm_out + 1, mm_out + 1, mm_out + 0],
                             [nn_out + 0, nn_out + 0, nn_out + 1, nn_out + 1]].T

            # Loop segments
            for ll, aa, bb in zip(svec, avec, bvec):
                zeta_mid = 0.5 * (zeta_panel_out[bb, :] + zeta_panel_out[aa, :])
                lv = zeta_panel_out[bb, :] - zeta_panel_out[aa, :]
                Lskew = algebra.skew((-Surf_out.rho * Surf_out.gamma[mm_out, nn_out]) * lv)

                # get vertices m,n indices
                mm_a, nn_a = mm_out + dmver[aa], nn_out + dnver[aa]
                mm_b, nn_b = mm_out + dmver[bb], nn_out + dnver[bb]
                # get vertices 1d index
                ii_a = [np.ravel_multi_index(
                    (cc, mm_a, nn_a), shape_fqs) for cc in range(3)]
                ii_b = [np.ravel_multi_index(
                    (cc, mm_b, nn_b), shape_fqs) for cc in range(3)]
                del mm_a, mm_b, nn_a, nn_b

                ### loop input surfaces coordinates
                for ss_in in range(n_surf):
                    ### Bound
                    Surf_in = Surfs[ss_in]
                    M_in_bound, N_in_bound = Surf_in.maps.M, Surf_in.maps.N
                    shape_zeta_in_bound = (3, M_in_bound + 1, N_in_bound + 1)
                    Dervert = Dervert_list[ss_out][ss_in]  # <- link
                    # deriv wrt induced velocity
                    dvind_mid, dvind_vert = dvinddzeta_cpp(
                        zeta_mid, Surf_in, is_bound=True, vortex_radius=Surf_in.vortex_radius)
                    # allocate coll
                    Df = np.dot(0.25 * Lskew, dvind_mid)
                    Dercoll[np.ix_(ii_a, ii_a)] += Df
                    Dercoll[np.ix_(ii_b, ii_a)] += Df
                    Dercoll[np.ix_(ii_a, ii_b)] += Df
                    Dercoll[np.ix_(ii_b, ii_b)] += Df
                    # allocate vert
                    Df = np.dot(0.5 * Lskew, dvind_vert)
                    Dervert[ii_a, :] += Df
                    Dervert[ii_b, :] += Df

                    ### wake
                    # deriv wrt induced velocity
                    dvind_mid, dvind_vert = dvinddzeta_cpp(
                        zeta_mid, Surfs_star[ss_in],
                        is_bound=False, vortex_radius=Surf_in.vortex_radius,
                        M_in_bound=Surf_in.maps.M)
                    # allocate coll
                    Df = np.dot(0.25 * Lskew, dvind_mid)
                    Dercoll[np.ix_(ii_a, ii_a)] += Df
                    Dercoll[np.ix_(ii_b, ii_a)] += Df
                    Dercoll[np.ix_(ii_a, ii_b)] += Df
                    Dercoll[np.ix_(ii_b, ii_b)] += Df

                    Df = np.dot(0.5 * Lskew, dvind_vert)
                    Dervert[ii_a, :] += Df
                    Dervert[ii_b, :] += Df

        # Loop output surf. TE
        # - we use Gammaw_0 over the TE
        # - we run along the positive direction as defined in the first row of
        # wake panels
        for nn_out in range(N_out):

            # get TE bound vertices m,n indices
            nn_a = nn_out + 1
            nn_b = nn_out

            # get segment and mid-point
            zeta_mid = 0.5 * (Surf_out.zeta[:, M_out, nn_b] + Surf_out.zeta[:, M_out, nn_a])
            lv = Surf_out.zeta[:, M_out, nn_b] - Surf_out.zeta[:, M_out, nn_a]
            Lskew = algebra.skew((-Surf_out.rho * Surfs_star[ss_out].gamma[0, nn_out]) * lv)

            # get vertices 1d index on bound
            ii_a = [np.ravel_multi_index(
                (cc, M_out, nn_a), shape_fqs) for cc in range(3)]
            ii_b = [np.ravel_multi_index(
                (cc, M_out, nn_b), shape_fqs) for cc in range(3)]

            ### loop input surfaces coordinates
            for ss_in in range(n_surf):
                ### Bound
                Surf_in = Surfs[ss_in]
                M_in_bound, N_in_bound = Surf_in.maps.M, Surf_in.maps.N
                shape_zeta_in_bound = (3, M_in_bound + 1, N_in_bound + 1)
                Dervert = Dervert_list[ss_out][ss_in]  # <- link
                # deriv wrt induced velocity
                dvind_mid, dvind_vert = dvinddzeta_cpp(zeta_mid, Surf_in,
                                                       is_bound=True,
                                                       vortex_radius=Surf_in.vortex_radius)
                # allocate coll
                Df = np.dot(0.25 * Lskew, dvind_mid)
                Dercoll[np.ix_(ii_a, ii_a)] += Df
                Dercoll[np.ix_(ii_b, ii_a)] += Df
                Dercoll[np.ix_(ii_a, ii_b)] += Df
                Dercoll[np.ix_(ii_b, ii_b)] += Df
                # allocate vert
                Df = np.dot(0.5 * Lskew, dvind_vert)
                Dervert[ii_a, :] += Df
                Dervert[ii_b, :] += Df

                ### wake
                # deriv wrt induced velocity
                dvind_mid, dvind_vert = dvinddzeta_cpp(
                    zeta_mid, Surfs_star[ss_in],
                    is_bound=False, vortex_radius=Surf_in.vortex_radius,
                    M_in_bound=Surf_in.maps.M)
                # allocate coll
                Df = np.dot(0.25 * Lskew, dvind_mid)
                Dercoll[np.ix_(ii_a, ii_a)] += Df
                Dercoll[np.ix_(ii_b, ii_a)] += Df
                Dercoll[np.ix_(ii_a, ii_b)] += Df
                Dercoll[np.ix_(ii_b, ii_b)] += Df

                # allocate vert
                Df = np.dot(0.5 * Lskew, dvind_vert)
                Dervert[ii_a, :] += Df
                Dervert[ii_b, :] += Df

    return Dercoll_list, Dervert_list


def dfunstdgamma_dot(Surfs):
    """
    Computes derivative of unsteady aerodynamic force with respect to changes in
    circulation.

    Note: the function also checks that the first derivative of the circulation
    at the linearisation point is null. If not, a further contribution to the
    added mass, depending on the changes in panel area and normal, arises and
    needs to be implemented.
    """

    DerList = []
    n_surf = len(Surfs)
    for ss in range(n_surf):
        Surf = Surfs[ss]

        ### check gamma_dot is zero
        assert (np.max(np.abs(Surf.gamma_dot)) < 1e-16), \
            'gamma_not not zero! Implement derivative w.r.t. lattice geometry changes'

        ### compute sensitivity
        wcv = Surf.get_panel_wcv()
        Kzeta = Surf.maps.Kzeta
        K = Surf.maps.K
        M, N = Surf.maps.M, Surf.maps.N
        shape_funst = (3, M + 1, N + 1)

        DerList.append(np.zeros((3 * Kzeta, K)))
        Der = DerList[-1]

        # loop panels (input, i.e. matrix columns)
        for pp in range(K):
            # get (m,n) indices of panel
            mm, nn = np.unravel_index(pp, (M, N))

            dfcoll = -Surf.rho * Surf.areas[mm, nn] * Surf.normals[:, mm, nn]

            for vv, dm, dn in zip(svec, dmver, dnver):
                df = wcv[vv] * dfcoll

                # get vertices 1d index
                iivec = [np.ravel_multi_index(
                    (vv, mm + dm, nn + dn), shape_funst) for vv in range(3)]

                Der[iivec, pp] += df

    return DerList


def wake_prop(MS, use_sparse=False, sparse_format='lil', settings=None):
    """
    Assembly of wake propagation matrices, in sparse or dense matrices format

    Note:
        Wake propagation matrices are very sparse. Nonetheless, allocation
        in dense format (from numpy.zeros) or sparse does not have important
        differences in terms of cpu time and memory used as numpy.zeros does
        not allocate memory until this is accessed

    Args:
        MS (MultiSurface): MultiSurface instance
        use_sparse (bool (optional)): Use sparse matrices
        sparse_format (str (optional)): Use either ``csc`` or ``lil`` format
        settings (dict (optional)): Dictionary with aerodynamic settings containing:
            cfl1 (bool): Defines if the wake shape complies with CFL=1
            dt (float): time step
    """

    try:
        cfl1 = settings['cfl1']
    except (KeyError, TypeError):
        # In case the key does not exist or settings=None
        cfl1 = True
    cout.cout_wrap("Computing wake propagation matrix with CFL1={}".format(cfl1), 1)

    n_surf = len(MS.Surfs)
    assert len(MS.Surfs_star) == n_surf, 'No. of wake and bound surfaces not matching!'

    dimensions = [None]*n_surf
    dimensions_star = [None]*n_surf

    for ss in range(n_surf):

        Surf = MS.Surfs[ss]
        Surf_star = MS.Surfs_star[ss]

        N, M, K = Surf.maps.N, Surf.maps.M, Surf.maps.K
        M_star, K_star = Surf_star.maps.M, Surf_star.maps.K
        assert Surf_star.maps.N == N, \
            'Bound and wake surface do not have the same spanwise discretisation'

        dimensions[ss] = [M, N, K]
        dimensions_star[ss] = [M_star, N, K_star]

        if not cfl1:
            # allocate...
            if use_sparse:
                if sparse_format == 'csc':
                    C = libsp.csc_matrix((K_star, K))
                    C_star = libsp.csc_matrix((K_star, K_star))
                elif sparse_format == 'lil':
                    C = sparse.lil_matrix((K_star, K))
                    C_star = sparse.lil_matrix((K_star, K_star))
            else:
                C = np.zeros((K_star, K))
                C_star = np.zeros((K_star, K_star))

            C_list = []
            Cstar_list = []

            # Compute flow velocity at wake
            uext = [np.zeros((3,
                             dimensions_star[ss][0],
                             dimensions_star[ss][1]))]

            try:
                Surf_star.zetac
            except AttributeError:
                Surf_star.generate_collocations()
            # Compute induced velocities in the wake
            Surf_star.u_ind_coll = np.zeros((3, M_star, N))

            for iin in range(N):
                # propagation from trailing edge
                conv_vec = Surf_star.zetac[:, 0, iin] - Surf.zetac[:, -1, iin]
                dist = np.linalg.norm(conv_vec)
                conv_dir_te = conv_vec/dist
                vel = Surf.u_input_coll[:, -1, iin]
                vel_value = np.dot(vel, conv_dir_te)
                cfl = settings['dt']*vel_value/dist

                C[iin, N * (M - 1) + iin] = cfl
                C_star[iin, iin] = 1.0 - cfl

                # wake propagation
                for mm in range(1, M_star):
                    conv_vec = Surf_star.zetac[:, mm, iin] - Surf_star.zetac[:, mm - 1, iin]
                    dist = np.linalg.norm(conv_vec)
                    conv_dir = conv_vec/dist
                    cfl = settings['dt']*vel_value/dist

                    C_star[mm * N + iin, (mm - 1) * N + iin] = cfl
                    C_star[mm * N + iin, mm * N + iin] = 1.0 - cfl

            C_list.append(C)
            Cstar_list.append(C_star)

    if cfl1:
        C_list, Cstar_list = wake_prop_from_dimensions(dimensions,
                                                       dimensions_star,
                                                       use_sparse=use_sparse,
                                                       sparse_format=sparse_format)

    return C_list, Cstar_list


def wake_prop_from_dimensions(dimensions, dimensions_star, use_sparse=False, sparse_format='lil'):
    """
    Same as ``wake_prop'' but using the dimensions directly
    """

    C_list = []
    Cstar_list = []

    # dimensions = [None]*n_surf

    n_surf = len(dimensions)
    assert len(dimensions_star) == n_surf, 'No. of wake and bound surfaces not matching!'

    for ss in range(n_surf):

        M, N, K = dimensions[ss]
        M_star, N_star, K_star = dimensions_star[ss]
        assert N_star == N, \
            'Bound and wake surface do not have the same spanwise discretisation'

        # allocate...
        if use_sparse:
            if sparse_format == 'csc':
                C = libsp.csc_matrix((K_star, K))
                C_star = libsp.csc_matrix((K_star, K_star))
            elif sparse_format == 'lil':
                C = sparse.lil_matrix((K_star, K))
                C_star = sparse.lil_matrix((K_star, K_star))
        else:
            C = np.zeros((K_star, K))
            C_star = np.zeros((K_star, K_star))

        # ... and fill
        iivec = np.array(range(N), dtype=int)
        # propagation from trailing edge
        C[iivec, N * (M - 1) + iivec] = 1.0
        # wake propagation
        for mm in range(1, M_star):
            C_star[mm * N + iivec, (mm - 1) * N + iivec] = 1.0

        C_list.append(C)
        Cstar_list.append(C_star)

    return C_list, Cstar_list


def test_wake_prop_term(M, N, M_star, N_star, use_sparse, sparse_format='csc'):
    """
    Test allocation of single term of wake propagation matrix
    """

    K = M * N
    K_star = M_star * N_star

    iivec = np.array(range(N), dtype=int)

    if use_sparse:
        if sparse_format == 'csc':
            C = sparse.csc_matrix((K_star, K))
            C_star = sparse.csc_matrix((K_star, K_star))
    else:
        C = np.zeros((K_star, K))
        C_star = np.zeros((K_star, K_star))

    ### Propagation from trailing edge
    C[iivec, N * (M - 1) + iivec] = 1.0

    ### wake propagation
    for mm in range(1, M_star):
        C_star[mm * N + iivec, (mm - 1) * N + iivec] = 1.0

    return C, C_star


if __name__ == '__main__':
    import time

    M, N = 20, 30
    M_star, N_star = M * 20, N

    t0 = time.time()
    C, C_star = test_wake_prop_term(M, N, M_star, N_star, use_sparse=False)
    tf = time.time() - t0
    print('Dense propagation matrix allocated in %.4f sec' % tf)

    t0 = time.time()
    Csp, Csp_star = test_wake_prop_term(M, N, M_star, N_star, use_sparse=True)
    tf = time.time() - t0
    print('csc sparse propagation matrix allocated in %.4f sec' % tf)

    # cProfile.runctx('self.assemble()', globals(), locals(), filename=self.prof_out)
    # 1/0

    ### compare sparse types
    Nx = 3000
    N1, N2 = 1000, Nx - 2000

    t0 = time.time()
    Z = sparse.csc_matrix((Nx, Nx))
    Z[:N1, :N1] = 2.
    Z[:N1, N1:] = 3.
    tfin = time.time() - t0
    print('csc allocated in %.6f sec' % tfin)

    t0 = time.time()
    Z = sparse.lil_matrix((Nx, Nx))
    Z[:N1, :N1] = 2.
    Z[:N1, N1:] = 3.
    Z = Z.tocsc()
    tfin = time.time() - t0
    print('lil->csc allocated in %.6f sec' % tfin)
