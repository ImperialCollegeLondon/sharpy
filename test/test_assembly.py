'''
Test assembly
S. Maraniello, 29 May 2018
'''

import numpy as np 
import warnings
import unittest
import matplotlib.pyplot as plt 

import sys, os
try:
	sys.path.append(os.environ['DIRuvlm3d'])
except KeyError:
	sys.path.append(os.path.abspath('../src/'))
import read, assembly, multisurfaces, libuvlm
from IPython import embed


class Test_assembly(unittest.TestCase):
	'''
	Test methods into assembly module
	'''

	def setUp(self):

		# select test case
		fname='./h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
		haero=read.h5file(fname)
		tsdata=haero.ts00000

		MS=multisurfaces.MultiAeroGridSurfaces(tsdata)
		MS.get_normal_ind_velocities_at_collocation_points()
		MS.verify_non_penetration()
		MS.verify_aic_coll()
		self.MS=MS



	def test_dAIC_dzeta_const_normals_self_bound(self):
		'''
		Test AIC derivative w.r.t. zeta assuming constant normals. Only the 
		contribution of the bound wake onto itself is checked.
		'''
		print('---- Testing assembly.dAIC_dzeta (constant normals. self bound)')
		
		MS=self.MS
		ss=0
		Surf_in=MS.Surfs[0]#MS.Surfs_star[0]##
		Surf_out=MS.Surfs[1]
		dAICcoll_an, dAICvert_an=assembly.dAICsdzeta_coll_Sin_to_Sout(
															   Surf_in,Surf_out)

		# map bound surface panel -> vertices
		Surf_out.maps.map_panels_to_vertices_1D_scalar()
		Surf_in.maps.map_panels_to_vertices_1D_scalar()
		K_out=Surf_out.maps.K
		Kzeta_out=Surf_out.maps.Kzeta 	
		K_in=Surf_in.maps.K
		Kzeta_in=Surf_in.maps.Kzeta 	
		
		# Reference induced velocity
		Uind0_norm=Surf_in.get_induced_velocity_over_surface(
									 Surf_out,target='collocation',Project=True)


		# --------------------------- derivative w.r.t. collocaiton points only

		step=1e-6
		dAICcoll_num=np.zeros((K_out,3*Kzeta_out))

		# loop collocation points
		for cc_out in range(K_out):
			# get panel coordinates
			mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
			nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
			# get normal
			uind0_norm=Uind0_norm[mm_out,nn_out]
			nc=Surf_out.normals[:,mm_out,nn_out]
			# get panel vertices
			zeta_panel=Surf_out.get_panel_vertices_coords(mm_out,nn_out)
			# get indices of panel vertices
			gl_ind_panel_out=Surf_out.maps.Mpv1d_scalar[cc_out]

			# perturb each panel vertices
			for vv in range(4):
				gl_ind_vv=gl_ind_panel_out[vv]

				# perturb each vertex component
				for vv_comp in range(3):
					zeta_panel_pert=zeta_panel.copy()
					zeta_panel_pert[vv,vv_comp]+=step

					# compute new collocation point
					zetac_pert=Surf_out.get_panel_collocation(zeta_panel_pert)

					# recompute induced velocity at collocation point
					# this step considers all nodes, we could instead only use libuvlm
					# but in this case we need to loop for surf_in panels
					uind=Surf_in.get_induced_velocity(zetac_pert)
					uind_norm=np.dot(nc,uind)
					#embed()
					# compute derivative/allocate
					dAICcoll_num[cc_out,gl_ind_vv+vv_comp*Kzeta_out]=\
													 (uind_norm-uind0_norm)/step

		er_coll=np.max(np.abs(dAICcoll_an-dAICcoll_num))


		# ----------------------------------- derivative w.r.t. vertices points
		dAICvert_num=np.zeros((K_out,3*Kzeta_in))

		# loop collocation points
		for cc_out in range(K_out):
			# get panel coordinates
			mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
			nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
			# get normal
			nc=Surf_out.normals[:,mm_out,nn_out]
			zetac=Surf_out.zetac[:,mm_out,nn_out]

			# perturb panels Surf_in
			for pp_in in range(K_in):
				# get (m,n) indices of panel
				mm_in=Surf_in.maps.ind_2d_pan_scal[0][pp_in]
				nn_in=Surf_in.maps.ind_2d_pan_scal[1][pp_in]	
				# get vertices coords and circulation			
				zeta_panel=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
				gamma_panel=Surf_in.gamma[mm_in,nn_in]
				# get vertices coordinates
				gl_ind_panel_in=Surf_in.maps.Mpv1d_scalar[pp_in]

				# compute panel original contribution to Uind
				uind0=libuvlm.biot_panel(zetac,zeta_panel,gamma=gamma_panel)
				uind0_norm=np.dot(uind0,nc)
				# perturb each panel vertices
				for vv in range(4):
					gl_ind_vv=gl_ind_panel_in[vv]

					# perturb each vertex component
					for vv_comp in range(3):
						zeta_panel_pert=zeta_panel.copy()
						zeta_panel_pert[vv,vv_comp]+=step

						# compute new induced velocity at collocaiton point
						uind=libuvlm.biot_panel(zetac,zeta_panel_pert,gamma=gamma_panel)
						uind_norm=np.dot(nc,uind)
						#embed()
						# compute derivative/allocate
						dAICvert_num[cc_out,gl_ind_vv+vv_comp*Kzeta_in]+=\
														 (uind_norm-uind0_norm)/step

		er_vert=np.max(np.abs(dAICvert_an-dAICvert_num))
		embed()












	# def test_dAIC_dzeta_coll_const_normals(self):

	# 	print('----------- Testing assembly.dAIC_dzeta_coll (constant normals)')
		
	# 	MS=self.MS

	# 	### compute dAIC_dzeta matrices
	# 	dAICcoll_an,dAICvert_an,dAICvert_star_an=\
	# 							assembly.dAICsdzeta_coll(MS.Surfs,MS.Surfs_star)

	
	# 	step=1e-4

	# 	# --------------------------- derivative w.r.t. collocaiton points only
	# 	dAICcoll_an_tot=[]
	# 	for ss_out in range(MS.n_surf):
	# 		dAICcoll_an_tot_here=dAICcoll_an[ss_out][0]
	# 		for ss_in in range(1,MS.n_surf):
	# 			dAICcoll_an_tot_here+=dAICcoll_an[ss_out][ss_in]
	# 		dAICcoll_an_tot.append(dAICcoll_an_tot_here)


	# 	dAICcoll_num_tot=[]
	# 	for ss_out in range(MS.n_surf):
	# 		Surf_out=MS.Surfs[ss_out]
	# 		K_out=Surf_out.maps.K
	# 		Kzeta_out=Surf_out.maps.Kzeta 
	# 		dAICcoll_num_tot_here=np.zeros((K_out,3*Kzeta_out))

	# 		# Reference induced velocity
	# 		Uind0=Surf_out.u_ind_coll_norm.copy()

	# 		# loop collocation points
	# 		for cc_out in range(K_out):
	# 			# get panel coordinates
	# 			mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
	# 			nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
	# 			# get collocation point and normal
	# 			zetac=Surf_out.zetac[:,mm_out,nn_out]
	# 			nc=Surf_out.normals[:,mm_out,nn_out]
	# 			uind0_norm=Surf_out.u_ind_coll_norm[mm_out,nn_out]

	# 			### perturb collocation point component
	# 			### ps: all surfaces are included
	# 			for ii in range(3):
	# 				uind=np.zeros((3,))
	# 				dzeta=np.zeros((3,))
	# 				dzeta[ii]=step
	# 				for ss_in in range(MS.n_surf):
	# 					Surf_in=MS.Surfs[ss_in]
	# 					Surf_in_star=MS.Surfs_star[ss_in]
	# 					uind+=Surf_in.get_induced_velocity(zetac+dzeta)
	# 					uind+=Surf_in_star.get_induced_velocity(zetac+dzeta)
	# 				uind_norm=np.dot(nc,uind)
	# 				dAICcoll_num_tot_here[cc_out,ii*K_out]=\
	# 												 (uind_norm-uind0_norm)/step

	# 		dAICcoll_num_tot.append(dAICcoll_num_tot_here)


	# 	embed()								   
	# 	# compare
	# 	Er=[]

	# 	for ss_out in range(MS.n_surf):
	# 		dAICcoll_an_tot=dAICcoll_an[ss_out][0]
	# 		for ss_in in range(1,MS.n_surf):
	# 			dAICcoll_an_tot=dAICcoll_an[ss_out][ss_in]
	# 		# compare vs numerical
	# 		Er.append(dAICcoll_an_tot - dAICcoll_num[ss_in] )


















	# def test_dWnvU_dzeta(self,PlotFlag=False):

	# 	print('---------------------------------- Testing assembly.dWnvU_dzeta')

	# 	Map,Surf=self.Map,self.Surf
	# 	tsdata=self.tsdata
	# 	ss=self.ss

	# 	# generate non-zero field of external force
	# 	u_ext0=tsdata.u_ext[ss]
	# 	u_ext0[0,:,:]=u_ext0[0,:,:]-20.0
	# 	u_ext0[1,:,:]=u_ext0[1,:,:]+60.0
	# 	u_ext0[2,:,:]=u_ext0[2,:,:]+30.0
	# 	u_ext0=u_ext0+np.random.rand(*u_ext0.shape)
	# 	Surf.u_ext=u_ext0

	# 	### analytical derivative
	# 	Surf.get_input_velocities_at_collocation_points()
	# 	Der=assembly.dWnvU_dzeta(Surf)

	# 	### numerical derivative
	# 	Surf.get_normal_input_velocities_at_collocation_points()
	# 	u_norm0=Surf.u_input_coll_norm.copy()
	# 	u_norm0_vec=u_norm0.reshape(-1,order='C')
	# 	zeta0=Surf.zeta
	# 	DerNum=np.zeros(Der.shape)

	# 	Steps=np.array([1e-2,1e-3,1e-4,1e-5,1e-6])
	# 	Er_max=0.0*Steps

	# 	for ss in range(len(Steps)):
	# 		step=Steps[ss]
	# 		for jj in range(3*Map.Kzeta):
	# 			# perturb
	# 			cc_pert=Map.ind_3d_vert_vect[0][jj]
	# 			mm_pert=Map.ind_3d_vert_vect[1][jj]
	# 			nn_pert=Map.ind_3d_vert_vect[2][jj]
	# 			zeta_pert=zeta0.copy()
	# 			zeta_pert[cc_pert,mm_pert,nn_pert]+=step
	# 			# calculate new normal velocity
	# 			Surf_pert=surface.AeroGridSurface(Map,zeta=zeta_pert,u_ext=u_ext0,
	# 												 gamma=tsdata.gamma[self.ss])
	# 			Surf_pert.get_normal_input_velocities_at_collocation_points()
	# 			u_norm_vec=Surf_pert.u_input_coll_norm.reshape(-1,order='C')
	# 			# FD derivative
	# 			DerNum[:,jj]=(u_norm_vec-u_norm0_vec)/step

	# 		er_max=np.max(np.abs(Der-DerNum))
	# 		print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
	# 		assert er_max<5e1*step, 'Error larger than 50 times step size'
	# 		Er_max[ss]=er_max

	# 	# assert error decreases with step size
	# 	for ss in range(1,len(Steps)):
	# 		assert Er_max[ss]<Er_max[ss-1],\
	# 		                   'Error not decreasing as FD step size is reduced'
	# 	print('------------------------------------------------------------ OK')

	# 	if PlotFlag:
	# 		fig = plt.figure('Spy Der',figsize=(10,4))
	# 		ax1 = fig.add_subplot(121)
	# 		ax1.spy(Der,precision=step)
	# 		ax2 = fig.add_subplot(122)
	# 		ax2.spy(DerNum,precision=step)
	# 		plt.show()


		




		

if __name__=='__main__':

	#unittest.main()
	T=Test_assembly()
	T.setUp()

	T.test_dAIC_dzeta_const_normals_self_bound()
	#T.test_dAIC_dzeta_coll_const_normals()

	## Induced velocity
	#T.test_dWnvU_dzeta()









