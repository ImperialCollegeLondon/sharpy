"""
Defines interpolation methods (geometrically-exact) and matrices (linearisation)
S. Maraniello, 20 May 2018
"""

import numpy as np



# -------------------------------------------------- interp at ollocation point

def get_panel_wcv(aM=0.5,aN=0.5):
	"""
	Produces a compact array with weights for bilinear interpolation, where
	aN,aM in [0,1] are distances in the chordwise and spanwise directions 
	such that:
		- (aM,aN)=(0,0) --> quantity at vertex 0
		- (aM,aN)=(1,0) --> quantity at vertex 1
		- (aM,aN)=(1,1) --> quantity at vertex 2
		- (aM,aN)=(0,1) --> quantity at vertex 3
	"""

	wcv=np.array([ (1-aM)*(1-aN), aM*(1-aN), aM*aN, aN*(1-aM) ])

	return wcv 


def get_Wvc_scalar(Map,aM=0.5,aN=0.5):
	"""
	Produce scalar interpolation matrix Wvc for state-space realisation.

	Important: this will not work for coordinates extrapolation, as it would
	require information about the panel size. It works for other forces/scalar 
	quantities extrapolation. It assumes the quantity at the collocation point
	is determined proportionally to the weight associated to each vertex and 
	obtained through get_panel_wcv. 
	"""

	# initialise
	K,Kzeta=Map.K,Map.Kzeta
	Wvc=np.zeros((Kzeta,K))

	# retrieve weights
	wcv=get_panel_wcv(aM,aN)
	wvc=wcv.T

	# define mapping panels to vertices (for scalars)
	if not hasattr(Map,'Mvp1d_scalar'):
		Map.map_panels_to_vertices_1D_scalar()

	# loop through panels
	for jj in range(K):
		# loop through local vertex index
		for vv in range(4):
			# vertex 1d index
			ii = Map.Mpv1d_scalar[jj,vv]
			# allocate
			Wvc[ii,jj]=wvc[vv]

	return Wvc


def get_Wvc_vector(Wvc_scalar):

	Kzeta,K=Wvc_scalar.shape
	Wvc=np.zeros((3*Kzeta,3*K))

	for cc in range(3):
		Wvc[cc*Kzeta:(cc+1)*Kzeta,cc*K:(cc+1)*K]=Wvc_scalar 

	return Wvc 


def get_Wnv_vector(SurfGeo,aM=0.5,aN=0.5):
	"""
	Provide projection matrix from nodal velocities to normal velocity at 
	collocation points
	"""

	# initialise
	Map=SurfGeo.maps
	K,Kzeta=Map.K,Map.Kzeta

	# retrieve scaling matrix
	Wvc_scalar=get_Wvc_scalar(Map,aM,aN)
	Wvc=get_Wvc_vector(Wvc_scalar)
	Wcv=Wvc.T
	del Wvc_scalar, Wvc
	
	# Build Wnc matrix
	Nmat=SurfGeo.normals.reshape((3,K))
	Wnc=np.concatenate([
		     np.diag(Nmat[0,:]),np.diag(Nmat[1,:]),np.diag(Nmat[2,:]) ], axis=1) 
	
	Wnv=np.dot(Wnc,Wcv)

	return Wnv





# -----------------------------------------------------------------------------

#
# if __name__=='__main__':
#
# 	import read
# 	import gridmapping
# 	import surface
# 	import matplotlib.pyplot as plt
#
# 	# select test case
# 	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
# 	haero=read.h5file(fname)
# 	tsdata=haero.ts00000
#
# 	# select surface and retrieve data
# 	ss=0
# 	M,N=tsdata.dimensions[ss]
# 	Map=gridmapping.AeroGridMap(M,N)
# 	SurfGeo=surface.AeroGridGeo(Map,tsdata.zeta[ss])
#
# 	# generate geometry data
# 	SurfGeo.generate_areas()
# 	SurfGeo.generate_normals()
# 	#SurfGeo.aM,SurfGeo.aN=0.25,0.75
# 	SurfGeo.generate_collocations()
#
#
# 	# ---------------------------------------------------------------- Test Wvc
# 	zeta_vec=SurfGeo.zeta.reshape(-1,order='C')
# 	Wvc_scalar=get_Wvc_scalar(Map)
# 	Wvc=get_Wvc_vector(Wvc_scalar)
# 	zetac_vec=np.dot(Wvc.T,zeta_vec)
# 	zetac=zetac_vec.reshape(Map.shape_pan_vect)
# 	SurfGeo.plot(plot_normals=False)
# 	SurfGeo.ax.scatter(zetac[0],zetac[1],zetac[2],zdir='z',s=6,c='b',marker='+')
# 	# # way back - can't work
# 	# zetav_vec=np.dot(Wvc,zetac_vec)
# 	# zetav=zetav_vec.reshape(Map.shape_vert_vect)
# 	# SurfGeo.ax.scatter(zetav[0],zetav[1],zetav[2],zdir='z',s=6,c='k',marker='+')
# 	# plt.show()
# 	plt.close('all')
#
#
# 	# ---------------------------------------------------------------- Test wnv
# 	# generate non-zero field of external force
# 	u_ext=tsdata.u_ext[ss]
# 	u_ext[0,:,:]=u_ext[0,:,:]-20.0
# 	u_ext[1,:,:]=u_ext[1,:,:]+60.0
# 	u_ext[2,:,:]=u_ext[2,:,:]+30.0
# 	u_ext=u_ext+np.random.rand(*u_ext.shape)
# 	# interpolate velocity at collocation points
#
# 	# compute normal velocity at panels
# 	wcv=get_panel_wcv(aM=0.5,aN=0.5)
# 	u_norm=np.zeros((M,N))
# 	for mm in range(M):
# 		for nn in range(N):
# 			# get velocity at panel corners
# 			mpv=SurfGeo.maps.from_panel_to_vertices(mm,nn)
# 			uc=np.zeros((3,))
# 			for vv in range(4):
# 				uc=uc+wcv[vv]*u_ext[:,mpv[vv,0],mpv[vv,1]]
# 			u_norm[mm,nn]=np.dot(uc,SurfGeo.normals[:,mm,nn])
# 	u_norm_vec_ref=u_norm.reshape(-1,order='C')
# 	# compute normal velocity through projection matrix
# 	u_ext_vec=u_ext.reshape(-1,order='C')
# 	Wnv=get_Wnv_vector(SurfGeo)
# 	u_norm_vec=np.dot(Wnv,u_ext_vec)
#
# 	assert np.max(np.abs(u_norm_vec-u_norm_vec_ref))<1e-12, 'Wnv not correct'
