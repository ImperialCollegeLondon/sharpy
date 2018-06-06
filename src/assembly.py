'''
Linearise UVLM assembly
S. Maraniello, 25 May 2018
'''

import numpy as np
#import multisurfaces

import libder.dWncUc_dzeta
import libder.dbiot

from IPython import embed







def AICs(Surfs,Surfs_star,target='collocation',Project=True):
	'''
	Given a list of bound (Surfs) and wake (Surfs_star) instances of 
	surface.AeroGridSurface, returns the list of AIC matrices in the format:
	 	- AIC_list[ii][jj] contains the AIC from the bound surface Surfs[jj] to 
	 	Surfs[ii].
	 	- AIC_star_list[ii][jj] contains the AIC from the wake surface Surfs[jj] 
	 	to Surfs[ii].
	'''

	AIC_list=[]
	AIC_star_list=[]

	n_surf=len(Surfs)
	assert len(Surfs_star)==n_surf,\
							   'Number of bound and wake surfaces much be equal'

	for ss_out in range(n_surf):
		AIC_list_here=[]
		AIC_star_list_here=[]
		Surf_out=Surfs[ss_out]

		for ss_in in range(n_surf):
			# Bound surface
			Surf_in=Surfs[ss_in]
			AIC_list_here.append(Surf_in.get_aic_over_surface(
										Surf_out,target=target,Project=Project))
			# Wakes
			Surf_in=Surfs_star[ss_in]
			AIC_star_list_here.append(Surf_in.get_aic_over_surface(
										Surf_out,target=target,Project=Project))
		AIC_list.append(AIC_list_here)
		AIC_star_list.append(AIC_star_list_here)	

	return AIC_list, AIC_star_list








def dAICsdzeta_coll_Sin_to_Sout(Surf_in,Surf_out):
	'''
	Computes derivative matrix of
		nc*dQ/dzeta
	where Q is the induced velocity induced by Surf_in onto Surf_out. The panel
	normals of Surf_out are constant. 
	'''

	# calc collocation points (and weights)
	if not hasattr(Surf_out,'zetac'):
		Surf_out.generate_collocations()
	ZetaColl=Surf_out.zetac
	wcv_out=Surf_out.get_panel_wcv()

	# allocate matrices
	K_out=Surf_out.maps.K
	Kzeta_out=Surf_out.maps.Kzeta
	K_in=Surf_in.maps.K
	Kzeta_in=Surf_in.maps.Kzeta
	Der_coll=np.zeros((K_out,3*Kzeta_out))
	Der_vert=np.zeros((K_out,3*Kzeta_in))

	# create mapping panels to vertices to loop 
	Surf_out.maps.map_panels_to_vertices_1D_scalar()
	Surf_in.maps.map_panels_to_vertices_1D_scalar()


	##### loop collocation points
	for cc_out in range(K_out):

		# get (m,n) indices of collocation point
		mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
		nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
		# get coords and normal
		zetac_here=ZetaColl[:,mm_out,nn_out]
		nc_here=Surf_out.normals[:,mm_out,nn_out]
		# get indices of panel vertices
		gl_ind_panel_out=Surf_out.maps.Mpv1d_scalar[cc_out]


		######  loop panels input surface 
		for pp_in in range(K_in):
			# get (m,n) indices of panel
			mm_in=Surf_in.maps.ind_2d_pan_scal[0][pp_in]
			nn_in=Surf_in.maps.ind_2d_pan_scal[1][pp_in]	
			# get vertices coords and circulation			
			zeta_panel=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
			gamma_panel=Surf_in.gamma[mm_in,nn_in]

			# get local derivatives
			der_zetac,der_zeta_panel=libder.dbiot.eval_panel(
									zetac_here,zeta_panel,gamma_pan=gamma_panel)
			
			
			### Allocate collocation point contribution
			der_zetac_proj=np.dot(nc_here,der_zetac)
			for vv in range(4):
				gl_ind_vv=gl_ind_panel_out[vv]
				for vv_comp in range(3):
					Der_coll[cc_out,gl_ind_vv+vv_comp*Kzeta_out]+=\
											 wcv_out[vv]*der_zetac_proj[vv_comp]

			### Allocate panel vertices contributions
			# get global indices of panel vertices
			gl_ind_panel_in=Surf_in.maps.Mpv1d_scalar[pp_in]
			for vv in range(4):
				gl_ind_vv=gl_ind_panel_in[vv]
				for vv_comp in range(3):
					Der_vert[cc_out,gl_ind_vv+vv_comp*Kzeta_in]+=\
									np.dot(nc_here,der_zeta_panel[vv,vv_comp,:])

	return Der_coll, Der_vert




################################################################################






def dAICsdzeta_coll(Surfs,Surfs_star):
	'''
	Produces a list of derivative matrix d(AIC*Gamma)/dzeta, where AIC are the 
	influence coefficient matrices at the bound surfaces collocation point, 
	ASSUMING constant panel norm.
	'''

	dAICcoll_list=[]
	dAICvert_list=[]
	dAICcoll_star_list=[]	
	dAICvert_star_list=[]

	n_surf=len(Surfs)
	assert len(Surfs_star)==n_surf,\
							   'Number of bound and wake surfaces much be equal'

	for ss_out in range(n_surf):
		dAICcoll_list_here=[]
		dAICvert_list_here=[]
		dAICcoll_star_list_here=[]
		dAICvert_star_list_here=[]

		Surf_out=Surfs[ss_out]
		K_out=Surf_out.maps.K
		Kzeta_out=Surf_out.maps.Kzeta

		if not hasattr(Surf_out,'zetac'):
			Surf_out.generate_collocations()
		ZetaColl=Surf_out.zetac

		for ss_in in range(n_surf):
			
			##### ------------------------------- bound
			Surf_in=Surfs[ss_in]
			K_in=Surf_in.maps.K
			Kzeta_in=Surf_in.maps.Kzeta
			# create mapping panels to vertices
			Surf_in.maps.map_panels_to_vertices_1D_scalar()
			# allocate matrices
			daic_coll=np.zeros((K_out,3*Kzeta_out))
			daic_vert=np.zeros((K_out,3*Kzeta_in))

			# loop collocation points
			for cc_out in range(K_out):

				# get coord_collocation point
				mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
				nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
				zetac_here=ZetaColl[:,mm_out,nn_out]
				# get normal to panel
				nc_here=Surf_out.normals[:,mm_out,nn_out]
				# get global indices of zetac_here
				ind_zetac=[cc_out+ii*Kzeta_out for ii in range(3)]
				
				# loop panels
				for pp_in in range(K_in):
					# get panel indices
					mm_in=Surf_in.maps.ind_2d_pan_scal[0][pp_in]
					nn_in=Surf_in.maps.ind_2d_pan_scal[1][pp_in]	
					#  get coord_and circulation			
					zeta_panel=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
					gamma_panel=Surf_in.gamma[mm_in,nn_in]

					# get derivatives
					# try:
					der_zetac,der_zeta_panel=libder.dbiot.eval_panel(
									zetac_here,zeta_panel,gamma_pan=gamma_panel)
					# except TypeError:
					# 	embed()

					# get global indices of panel vertices
					ind_zeta_panel=Surf_in.maps.Mpv1d_scalar[pp_in]

					# allocate
					daic_coll[cc_out,ind_zetac]=np.dot(nc_here,der_zetac)
					for vv in range(4):
						for vv_comp in range(3):
							ind_zetavv_comp=ind_zeta_panel[vv]+vv_comp*Kzeta_in
							daic_vert[cc_out,ind_zetavv_comp]=\
									np.dot(nc_here,der_zeta_panel[vv,vv_comp,:])

			dAICcoll_list_here.append(daic_coll)
			dAICvert_list_here.append(daic_vert)

			##### ------------------------------- wake
			Surf_in=Surfs_star[ss_in]
			K_in=Surf_in.maps.K
			Kzeta_in=Surf_in.maps.Kzeta
			# create mapping panels to vertices
			Surf_in.maps.map_panels_to_vertices_1D_scalar()
			# allocate matrices
			daic_coll=np.zeros((K_out,3*Kzeta_out))
			daic_vert=np.zeros((K_out,3*Kzeta_in))

			# loop collocation points
			for cc_out in range(K_out):

				# get coord_collocation point
				mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
				nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
				zetac_here=ZetaColl[:,mm_out,nn_out]
				# get normal to panel
				nc_here=Surf_out.normals[:,mm_out,nn_out]
				# get global indices of zetac_here
				ind_zetac=[cc_out+ii*Kzeta_out for ii in range(3)]
				
				# loop panels
				for pp_in in range(K_in):
					# get panel indices
					mm_in=Surf_in.maps.ind_2d_pan_scal[0][pp_in]
					nn_in=Surf_in.maps.ind_2d_pan_scal[1][pp_in]	
					#  get coord_and circulation			
					zeta_panel=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
					gamma_panel=Surf_in.gamma[mm_in,nn_in]

					# get derivatives
					der_zetac,der_zeta_panel=libder.dbiot.eval_panel(
									zetac_here,zeta_panel,gamma_pan=gamma_panel)

					# get global indices of panel vertices
					ind_zeta_panel=Surf_in.maps.Mpv1d_scalar[pp_in]

					# allocate
					daic_coll[cc_out,ind_zetac]=np.dot(nc_here,der_zetac)
					for vv in range(4):
						for vv_comp in range(3):
							ind_zetavv_comp=ind_zeta_panel[vv]+vv_comp*Kzeta_in
							daic_vert[cc_out,ind_zetavv_comp]=\
									np.dot(nc_here,der_zeta_panel[vv,vv_comp,:])

			dAICvert_star_list_here.append(daic_vert)
			dAICcoll_star_list_here.append(daic_coll)

		dAICcoll_list.append(dAICcoll_list_here)
		dAICvert_list.append(dAICvert_list_here)
		dAICcoll_star_list.append(dAICcoll_star_list_here)
		dAICvert_star_list.append(dAICvert_star_list_here)

	return dAICcoll_list, dAICvert_list, dAICcoll_star_list, dAICvert_star_list



def dWnvU_dzeta(Surf):
	'''
	Build derivative of Wnv*u_ext w.r.t grid coordinates. Input Surf can be:
	- an instance of surface.AeroGridSurface.
	- a list of instance of surface.AeroGridSurface. 	
	Refs:
	- develop_sym.linsum_Wnc
	- libder.dWncUc_dzeta
	'''

	if type(Surf) is list:
		n_surf=len(Surf)
		DerList=[]
		for ss in range(n_surf):
			DerList.append(dWncUc_dzeta(Surf[ss]))
			return DerList

	Map=Surf.maps
	K,Kzeta=Map.K,Map.Kzeta
	Der=np.zeros((K,3*Kzeta))

	# map panel to vertice
	if not hasattr(Map.Mpv,'Mpv1d_scalar'):
		Map.map_panels_to_vertices_1D_scalar()
	if not hasattr(Map.Mpv,'Mpv'):
		Map.map_panels_to_vertices()

	# map u_normal 2d to 1d
	# map_panels_1d_to_2d=np.unravel_index(range(K),
	# 						   				  dims=Map.shape_pan_scal,order='C')
	# for ii in range(K):

	for ii in Map.ind_1d_pan_scal:

		# panel m,n coordinates
		m_pan,n_pan=Map.ind_2d_pan_scal[0][ii],Map.ind_2d_pan_scal[1][ii]
		# extract u_input_coll
		u_input_coll_here=Surf.u_input_coll[:,m_pan,n_pan]

		# find vertices
		mpv=Map.Mpv[m_pan,n_pan,:,:]

		# extract m,n coordinates of vertices
		zeta00=Surf.zeta[:,mpv[0,0],mpv[0,1]]
		zeta01=Surf.zeta[:,mpv[1,0],mpv[1,1]]
		zeta02=Surf.zeta[:,mpv[2,0],mpv[2,1]]
		zeta03=Surf.zeta[:,mpv[3,0],mpv[3,1]]

		# calculate derivative
		Dlocal=libder.dWncUc_dzeta.eval(zeta00,zeta01,zeta02,zeta03,
															  u_input_coll_here)

		for vv in range(4):
			# find 1D position of vertices
			jj=Map.Mpv1d_scalar[ii,vv]

			# allocate derivatives
			Der[ii,jj]=Dlocal[vv,0] # w.r.t. x
			Der[ii,jj+Kzeta]=Dlocal[vv,1] # w.r.t. y
			Der[ii,jj+2*Kzeta]=Dlocal[vv,2] # w.r.t. z

	return Der








# -----------------------------------------------------------------------------


if __name__=='__main__':
	pass 
	# import read
	# import gridmapping, surface
	# import matplotlib.pyplot as plt 

	# # select test case
	# fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'

	# haero=read.h5file(fname)
	# tsdata=haero.ts00000

	# # select surface and retrieve data
	# ss=0
	# M,N=tsdata.dimensions[ss]
	# Map=gridmapping.AeroGridMap(M,N)
	# Surf=surface.AeroGridSurface(Map,tsdata.zeta[ss])








			









