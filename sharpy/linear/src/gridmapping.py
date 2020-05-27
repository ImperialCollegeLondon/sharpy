"""Mapping methods for bound surface panels

S. Maraniello, 19 May 2018
"""

import numpy as np
import sharpy.utils.cout_utils as cout


class AeroGridMap():
    """
    Produces mapping between panels, segment and vertices of a surface.
    Grid elements are identified through the indices (m,n), where:
        - m: chordwise index
        - n: spanwise index
    The same indexing is applied to panel, vertices and segments.

    Elements:
    - panels=(M,N)
    - vertices=(M+1,N+1)
    - segments: these are divided in segments developing along the chordwise and
    spanwise directions.
        - chordwise: (M,N+1)
        - spanwise: (M+1,N)

    Mapping structures:
    - Mpv: for each panel (mp,np) returns the chord/span-wise indices of its
    vertices, (mv,nv). This has size (M,N,4,2)
    - Mps: maps each panel (mp,np) to the ii-th segment. This has size (M,N,4,2)


    # - Mps_extr: for each panel (m,n) returns the indices of the extrema of each side
    # of the panel.

    Note:
    - mapping matrices are stored as np.int16 or np.int32 arrays
    """

    def __init__(self, M: 'number of chord-wise', N: 'number of span-wise'):
        ### init
        self.M = M
        self.N = N
        self.K = M * N  # panels number
        self.Kzeta = (M + 1) * (N + 1)  # vertices number

        ### format
        self.intxx = np.int16
        if self.Kzeta > np.iinfo(np.int16).max: self.intxx = np.int32

        ### shapes for multi-dimensional arrays
        self.shape_pan_scal = (M, N)
        self.shape_pan_vect = (3, M, N)
        self.shape_vert_scal = (M + 1, N + 1)
        self.shape_vert_vect = (3, M + 1, N + 1)

        # local mapping segment/vertices of a panel
        self.svec = np.array([0, 1, 2, 3], dtype=self.intxx)  # seg. number
        self.avec = np.array([0, 1, 2, 3], dtype=self.intxx)  # 1st vertex of seg.
        self.bvec = np.array([1, 2, 3, 0], dtype=self.intxx)  # 2nd vertex of seg.

        # deltas to convert panel (m,n) to vertices (m,n) indices. For each
        # vertex of the panel:
        # 	m_ver,n_ver = m+self.dmver, n+self.dnver
        self.dmver = np.array([0, 1, 1, 0], dtype=self.intxx)
        self.dnver = np.array([0, 0, 1, 1], dtype=self.intxx)

        ### variables 1D <-> nD mapping
        # # example:
        # A=np.random.rand(3,4,5)
        # a=A.reshape(-1,order='C')
        # Na=len(a)
        # ind_1d=range(Na)
        # ind_3d=np.unravel_index(ind_1d,A.shape)
        # ind_1d=np.ravel_multi_index(ind_3d,A.shape)
        # a[range(Na)]==a[ind_1d]==A[ind_3d]

        # vectors defined at vertices
        self.ind_1d_vert_vert = range(3 * self.Kzeta)
        self.ind_3d_vert_vect = np.unravel_index(self.ind_1d_vert_vert,
                                                 shape=self.shape_vert_vect, order='C')
        # scalars defined at panels
        self.ind_1d_pan_scal = range(self.K)
        self.ind_2d_pan_scal = np.unravel_index(self.ind_1d_pan_scal,
                                                shape=self.shape_pan_scal, order='C')

        # ### mapping to/from 1D arrays
        # self.maps.ind_vector_vertices_to_3d=
        # self.maps.ind_vector_vertices_to_1d=
        # self.maps.ind_scalar_panel_to_3d=
        # self.maps.ind_scalar_panel_to_1d=

    def map_all(self):
        self.map_panels_to_vertices()
        self.map_panels_to_segments()
        self.map_vertices_to_panels()

    # ------------------------------------------------------ panels to vertices

    def map_panels_to_vertices_1D_scalar(self):
        """
        Mapping:
        - FROM: the index of a scalar quantity defined at panel collocation point
        and stored in 1D array.
        - TO: index of a scalar quantity defined at vertices and stored in 1D

        The Mpv1d_scalar has size (K,4) where:
            [1d index of panel, index of vertex 0,1,2 or 3]
        """

        self.Mpv1d_scalar = np.zeros((self.K, 4), dtype=self.intxx)

        # Map: panels 1D -> 3d
        if not hasattr(self, 'Mpv'):
            self.map_panels_to_vertices()
        mn_panels = np.unravel_index(range(self.K),
                                     shape=self.shape_pan_scal, order='C')
        # Mpv_new=self.Mpv[mn_panels] # from k to vertices

        for kk in range(self.K):

            # map from kk-th panel to vertices (m,n)
            mpv = self.Mpv[mn_panels[0][kk], mn_panels[1][kk], :, :]

            # loop through vertices
            for vv in range(4):
                self.Mpv1d_scalar[kk, vv] = np.ravel_multi_index(mpv[vv, :],
                                                                 dims=self.shape_vert_scal, order='C')

    def map_panels_to_vertices(self):
        """
        Mapping from panel of vertices. self.Mpv is a (M,N,4,2) array such that
        its element are:
            [m, n, local_vertex_number, spanwise/chordwise indices of vertex]
        """

        M, N = self.M, self.N
        self.Mpv = np.zeros((M, N, 4, 2), dtype=self.intxx)

        for mm in range(M):
            for nn in range(N):
                self.Mpv[mm, nn, :, :] = self.from_panel_to_vertices(mm, nn)

    def from_panel_to_vertices(self, m: 'chordwise index', n: 'spanwise index'):
        """
        From panel of indices (m,n) to indices of vertices
        """

        # mpv=np.zeros((4,2),dtype=self.intxx)
        # mpv[0,:]=m  ,n
        # mpv[1,:]=m+1,n
        # mpv[2,:]=m+1,n+1
        # mpv[3,:]=m  ,n+1
        mpv = np.array([m + self.dmver, n + self.dnver]).T

        return mpv

    # ------------------------------------------------------ vertices to panels

    def map_vertices_to_panels_1D_scalar(self):
        """
        Mapping:
        - FROM: the index of a scalar quantity defined at vertices and stored in
        1D array.
        - TO: index of a scalar quantity defined at panels and stored in 1D

        The Mpv1d_scalar has size (Kzeta,4) where:
            [1d index of vertex, index of vertex 0,1,2 or 3 w.r.t. panel]
        """

        self.Mvp1d_scalar = np.zeros((self.Kzeta, 4), dtype=self.intxx)

        # Map: vertices 1D -> 3d
        if not hasattr(self, 'Mvp'):
            self.map_vertices_to_panels()
        mn_vertices = np.unravel_index(range(self.Kzeta),
                                       shape=self.shape_vert_scal, order='C')

        for kk in range(self.Kzeta):

            # map from kk-th vertex to panels (m,n)
            mvp = self.Mvp[mn_vertices[0][kk], mn_vertices[1][kk], :, :]

            # loop through vertex local order
            for vv in range(4):
                # check if vertex is vv-th for any panel
                if np.all(mvp[vv, :] != -1):
                    self.Mvp1d_scalar[kk, vv] = np.ravel_multi_index(mvp[vv, :],
                                                                     dims=self.shape_pan_scal, order='C')
                else:
                    self.Mvp1d_scalar[kk, vv] = -1

    def map_vertices_to_panels(self):
        """
        Maps from vertices to panels. Produces a (M+1,N+1,4,2) array, associating
        vertices to panels. Its elements are:
            [m vertex,
                n vertex,
                    vertex local index,
                        chordwise/spanwise panel indices]
        """

        M, N = self.M, self.N
        self.Mvp = np.zeros((M + 1, N + 1, 4, 2), dtype=self.intxx)

        for mm in range(M + 1):
            for nn in range(N + 1):
                self.Mvp[mm, nn, :, :] = self.from_vertex_to_panel(mm, nn)
                # remove out of grid panels
                mmvec_rem = self.Mvp[mm, nn, :, 0] >= M
                nnvec_rem = self.Mvp[mm, nn, :, 1] >= N
                self.Mvp[mm, nn, mmvec_rem, 0] = -1
                self.Mvp[mm, nn, nnvec_rem, 1] = -1

    def from_vertex_to_panel(self, m: 'chordwise index', n: 'spanwise index'):
        """
        Returns the panel for which the vertex is locally numbered as 0,1,2,3.
        Returns a (4,2) array such that its elements are:
            [vv_local,(m,n) of panel]
        where vv_local is the local verteix number.

        Important: indices -1 are possible is the vertex does not have local index
        0,1,2 or 3 with respect to any panel.
        """

        mvp = np.zeros((4, 2), dtype=self.intxx)
        mvp[0, :] = [m, n]
        mvp[1, :] = [m - 1, n]
        mvp[2, :] = [m - 1, n - 1]
        mvp[3, :] = [m, n - 1]

        return mvp

    # ------------------------------------------------------ panels to segments

    def map_panels_to_segments(self):
        """
        Mapping from panel of segments. self.Mpv is a (M,N,4,2) array such
        that:
            [m, n, local_segment_number,
                        chordwise/spanwise index of segment,]
        """

        M, N = self.M, self.N
        self.Mps = np.zeros((M, N, 4, 2), dtype=self.intxx)

        for mm in range(M):
            for nn in range(N):
                self.Mps[mm, nn, :, :] = self.from_panel_to_segments(mm, nn)

    def from_panel_to_segments(self, m: 'chordwise index', n: 'spanwise index'):
        """
        For each panel (m,n) it provides the ms,ns indices of each segment.
        """

        mps = np.zeros((4, 2), dtype=np.int32)
        mps[0, :] = m, n
        mps[1, :] = m + 1, n
        mps[2, :] = m, n + 1
        mps[3, :] = m, n

        return mps

    # # ---------------------------------------------- panels to segments extrema

    # def map_panels_to_segments(self):
    # 	"""
    # 	Mapping from panel of segments. self.Mpv is a (M,N,4,2,2) array such
    # 	that:
    # 		[m, n, local_segment_number,
    # 			   		spanwise/chordwise indices of vertex 0,
    # 			   			spanwise/chordwise indices of vertex 1]
    # 	"""

    # 	M,N=self.M,self.N
    # 	self.Mps=np.zeros((M,N,4,2,2),dtype=self.intxx)

    # 	for mm in range(M):
    # 		for nn in range(N):
    # 			self.Mps[mm,nn,:,:,:]=self.from_panel_to_segments(mm,nn)

    # def from_panel_to_segments(self,m:'chordwise index',n:'spanwise index'):
    # 	"""
    # 	For each panel (m,n) it provides the indices of the extrema of each
    # 	segment.
    # 		[segment number,indices extrema 0,indices extrema 1]
    # 	"""

    # 	mpv=self.from_panel_to_vertices(m,n)
    # 	mps=np.zeros((4,2,2),dtype=np.int32)
    # 	mps[0,0,:],mps[0,1,:]=mpv[0,:],mpv[1,:]
    # 	mps[1,0,:],mps[1,1,:]=mpv[1,:],mpv[2,:]
    # 	mps[2,0,:],mps[2,1,:]=mpv[2,:],mpv[3,:]
    # 	mps[3,0,:],mps[3,1,:]=mpv[3,:],mpv[0,:]

    # 	return mps


if __name__ == '__main__':
    M, N = 3, 5

    Map = AeroGridMap(M, N)

    ### multi-dimensional mapping
    Map.map_panels_to_vertices()
    Map.map_panels_to_segments()
    Map.map_vertices_to_panels()

    # 1D mappings
    Map.map_panels_to_vertices_1D_scalar()
    Map.map_vertices_to_panels_1D_scalar()
