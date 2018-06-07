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



	def test_dAIC_dzeta_const_normals_bound_to_bound(self):
		'''
		Test AIC derivative w.r.t. zeta assuming constant normals. Only the 
		contribution bound to bound surfaces contribution is verified.
		'''
		print('-- Testing assembly.dAIC_dzeta (constant normals, bound->bound)')
		
		MS=self.MS
		n_surf=MS.n_surf
		for ss_in in range(n_surf):
			for ss_out in range(n_surf):

				### bound to bound
				Surf_in=MS.Surfs[ss_in]
				Surf_out=MS.Surfs[ss_out]

				K_out=Surf_out.maps.K
				Kzeta_out=Surf_out.maps.Kzeta
				Kzeta_in=Surf_in.maps.Kzeta

				dAICcoll_an=np.zeros((K_out,3*Kzeta_out))
				dAICvert_an=np.zeros((K_out,3*Kzeta_in))		

				dAICcoll_an, dAICvert_an=\
						assembly.dAICsdzeta_coll_Sin_to_Sout(
										Surf_in,Surf_out,dAICcoll_an,
												dAICvert_an,Surf_in_bound=True)

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

				step=1e-7
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

				print('Bound\t%.2d->%.2d\tFDstep\tErColl\tErVert'%(ss_in,ss_out))
				print('\t\t%.1e\t%.1e\t%.1e\t' %(step,er_coll,er_vert))
				assert max(er_coll,er_vert)<1e2*step, 'Test failed!'





	def test_dAIC_dzeta_const_normals_wake_to_bound(self):
		'''
		Test AIC derivative w.r.t. zeta assuming constant normals. Only the 
		contribution of the wake to bound is checked.
		The derivatives w.r.t. bound collocaiton point are tested via FDs.
		'''
		print('-- Testing assembly.dAIC_dzeta (constant normals, wake->bound)')
		
		MS=self.MS
		n_surf=MS.n_surf
		for ss_in in range(n_surf):
			for ss_out in range(n_surf):

				### bound to bound
				Surf_in=MS.Surfs_star[ss_in]
				Surf_out=MS.Surfs[ss_out]
				K_out=Surf_out.maps.K
				Kzeta_out=Surf_out.maps.Kzeta
				Kzeta_bound_in=MS.Surfs[ss_in].maps.Kzeta #<--- not a mistake

				dAICcoll_an=np.zeros((K_out,3*Kzeta_out))
				dAICvert_an=np.zeros((K_out,3*Kzeta_bound_in))		
				# get bound influence
				dAICcoll_an, dAICvert_an=\
						assembly.dAICsdzeta_coll_Sin_to_Sout(
										Surf_in,Surf_out,dAICcoll_an,
												dAICvert_an,Surf_in_bound=False)



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


				# -------------------- derivative w.r.t. collocaiton points only

				step=1e-7
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
							# this step considers all nodes, we could instead only 
							# use libuvlm but in this case we need to loop for 
							# surf_in panels
							uind=Surf_in.get_induced_velocity(zetac_pert)
							uind_norm=np.dot(nc,uind)
							# compute derivative/allocate
							dAICcoll_num[cc_out,gl_ind_vv+vv_comp*Kzeta_out]=\
													 (uind_norm-uind0_norm)/step

				er_coll=np.max(np.abs(dAICcoll_an-dAICcoll_num))



				# ---------------------------- derivative w.r.t. vertices points

				N_in=Surf_in.maps.N
				Der_num=np.zeros((K_out,N_in+1,3,))

				# loop collocation points
				for cc_out in range(K_out):
					# get panel coordinates
					mm_out=Surf_out.maps.ind_2d_pan_scal[0][cc_out]
					nn_out=Surf_out.maps.ind_2d_pan_scal[1][cc_out]
					# get normal
					nc=Surf_out.normals[:,mm_out,nn_out]
					zetac=Surf_out.zetac[:,mm_out,nn_out]

					# perturb TE points
					# perturb panels Surf_in
					for pp_in in range(K_in):
						# get (m,n) indices of panel
						mm_in=Surf_in.maps.ind_2d_pan_scal[0][pp_in]
						nn_in=Surf_in.maps.ind_2d_pan_scal[1][pp_in]	

						if mm_in !=0:
							continue

						# get vertices coords and circulation			
						zeta_panel=Surf_in.get_panel_vertices_coords(mm_in,nn_in)
						gamma_panel=Surf_in.gamma[mm_in,nn_in]
						zeta00=zeta_panel[0,:]
						zeta03=zeta_panel[3,:]

						# compute panel original contribution to Uind
						uind0=libuvlm.biot_segment(zetac,zeta03,zeta00,gamma=gamma_panel)
						uind0_norm=np.dot(uind0,nc)

						# perturb each vertex component
						for vv_comp in range(3):
							### segment 00
							zeta00_pert=zeta00.copy()
							zeta00_pert[vv_comp]+=step
							uind=libuvlm.biot_segment(zetac,zeta03,zeta00_pert,
															  gamma=gamma_panel)
							uind_norm=np.dot(nc,uind)
							Der_num[cc_out,nn_in,vv_comp]+=(uind_norm-uind0_norm)/step

							### segment 03
							zeta03_pert=zeta03.copy()
							zeta03_pert[vv_comp]+=step
							uind=libuvlm.biot_segment(zetac,zeta03_pert,zeta00,
															  gamma=gamma_panel)
							uind_norm=np.dot(nc,uind)
							Der_num[cc_out,nn_in+1,vv_comp]+=(uind_norm-uind0_norm)/step


				### manually reshape output
				dAICvert_num=np.zeros((K_out,3*Kzeta_bound_in))
				M_bound_in=Kzeta_bound_in//(N_in+1)-1
				for comp in range(3):
					kstart=(1+comp)*Kzeta_bound_in-(N_in+1)
					kend=(1+comp)*Kzeta_bound_in 
					dAICvert_num[:,kstart:kend]=Der_num[:,:,comp]

				er_vert=np.max(np.abs(dAICvert_an-dAICvert_num))

				print('Wake%.2d->Bound%.2d\tFDstep\tErColl\tErVert'%(ss_in,ss_out))
				print('\t\t%.1e\t%.1e\t%.1e\t' %(step,er_coll,er_vert))
				assert max(er_coll,er_vert)<1e2*step, 'Test failed!'





















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
	T.test_dAIC_dzeta_const_normals_bound_to_bound()
	T.test_dAIC_dzeta_const_normals_wake_to_bound()
	

	## Induced velocity
	#T.test_dWnvU_dzeta()









