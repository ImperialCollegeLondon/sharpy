'''
Test assembly
S. Maraniello, 29 May 2018
'''

import numpy as np 
import warnings
import unittest
import itertools
import copy
import matplotlib.pyplot as plt 

import sys, os
try:
	sys.path.append(os.environ['DIRuvlm3d'])
except KeyError:
	sys.path.append(os.path.abspath('../src/'))
import read, assembly, multisurfaces, surface, libuvlm

from IPython import embed
np.set_printoptions(linewidth=200,precision=3)



def max_error_tensor(Pder_an,Pder_num):
	'''
	Finds the maximum error analytical derivatives Pder_an. The error is:
	- relative, if the element of Pder_an is nonzero
	- absolute, otherwise

	The function returns the absolute and relative error tensors, and the
	maximum error. 

	@warning: The relative error tensor may contain NaN or Inf if the
	analytical derivative is zero. These elements are filtered out during the
	search for maximum error, and absolute error is checked.
	'''

	Eabs=np.abs(Pder_num-Pder_an)
	Erel=np.abs(Eabs/Pder_an)

	# Relative error check: remove NaN and inf...
	iifinite=np.isfinite(Erel)
	err_max=0.0
	for err_here in Erel[iifinite]:
		if np.abs(err_here)>err_max:
			err_max=err_here

	# Zero elements check
	iinonzero=np.abs(Pder_an)<1e-15
	for der_here in Pder_num[iinonzero]:
		if np.abs(der_here)>err_max:
			err_max=der_here 

	return err_max, Eabs, Erel




class Test_assembly(unittest.TestCase):
	'''
	Test methods into assembly module
	'''

	def setUp(self):

		# select test case
		fname='./h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
		#fname='./h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
		haero=read.h5file(fname)
		tsdata=haero.ts00000

		MS=multisurfaces.MultiAeroGridSurfaces(tsdata)
		MS.get_normal_ind_velocities_at_collocation_points()
		MS.verify_non_penetration()
		MS.verify_aic_coll()
		MS.get_joukovski_qs()
		MS.verify_joukovski_qs()
		self.MS=MS


	def test_nc_dqcdzeta_bound_to_bound(self):
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
						assembly.nc_dqcdzeta_Sin_to_Sout(
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


				# ------------------- derivative w.r.t. collocaiton points only

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
							zetac_pert=Surf_out.get_panel_collocation(
																zeta_panel_pert)

							# recompute induced velocity at collocation point
							# this step considers all nodes, we could instead 
							# only use libuvlm but in this case we need to loop 
							# for surf_in panels
							uind=Surf_in.get_induced_velocity(zetac_pert)
							uind_norm=np.dot(nc,uind)
							#embed()
							# compute derivative/allocate
							dAICcoll_num[cc_out,gl_ind_vv+vv_comp*Kzeta_out]=\
													 (uind_norm-uind0_norm)/step

				er_coll=np.max(np.abs(dAICcoll_an-dAICcoll_num))


				# --------------------------- derivative w.r.t. vertices points
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
						uind0=libuvlm.biot_panel(zetac,zeta_panel,
															  gamma=gamma_panel)
						uind0_norm=np.dot(uind0,nc)
						# perturb each panel vertices
						for vv in range(4):
							gl_ind_vv=gl_ind_panel_in[vv]

							# perturb each vertex component
							for vv_comp in range(3):
								zeta_panel_pert=zeta_panel.copy()
								zeta_panel_pert[vv,vv_comp]+=step

								# compute new ind velocity at collocation pts
								uind=libuvlm.biot_panel(zetac,zeta_panel_pert,
															  gamma=gamma_panel)
								uind_norm=np.dot(nc,uind)
								#embed()
								# compute derivative/allocate
								dAICvert_num[cc_out,gl_ind_vv+vv_comp*Kzeta_in]+=\
													 (uind_norm-uind0_norm)/step

				er_vert=np.max(np.abs(dAICvert_an-dAICvert_num))

				print('Bound\t%.2d->%.2d\tFDstep\tErColl\tErVert'%(ss_in,ss_out))
				print('\t\t%.1e\t%.1e\t%.1e\t' %(step,er_coll,er_vert))
				assert max(er_coll,er_vert)<1e2*step, 'Test failed!'
			print('------------------------------------------------------------ OK')




	def test_nc_dqcdzeta_wake_to_bound(self):
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
						assembly.nc_dqcdzeta_Sin_to_Sout(
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
							zetac_pert=Surf_out.get_panel_collocation(
																zeta_panel_pert)

							# recompute induced velocity at collocation point
							# this step considers all nodes, we could instead 
							# only use libuvlm but in this case we need to loop 
							# for surf_in panels
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
						uind0=libuvlm.biot_segment(zetac,zeta03,zeta00,
															  gamma=gamma_panel)
						uind0_norm=np.dot(uind0,nc)

						# perturb each vertex component
						for vv_comp in range(3):
							### segment 00
							zeta00_pert=zeta00.copy()
							zeta00_pert[vv_comp]+=step
							uind=libuvlm.biot_segment(zetac,zeta03,zeta00_pert,
															  gamma=gamma_panel)
							uind_norm=np.dot(nc,uind)
							Der_num[cc_out,nn_in,vv_comp]+=\
													 (uind_norm-uind0_norm)/step

							### segment 03
							zeta03_pert=zeta03.copy()
							zeta03_pert[vv_comp]+=step
							uind=libuvlm.biot_segment(zetac,zeta03_pert,zeta00,
															  gamma=gamma_panel)
							uind_norm=np.dot(nc,uind)
							Der_num[cc_out,nn_in+1,vv_comp]+=\
													 (uind_norm-uind0_norm)/step


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
			print('------------------------------------------------------------ OK')



	def test_uc_dncdzeta(self,PlotFlag=False):

		print('---------------------------------- Testing assembly.uc_dncdzeta')

		MS=self.MS
		n_surf=MS.n_surf

		MS.get_ind_velocities_at_collocation_points()
		MS.get_normal_ind_velocities_at_collocation_points()

		for ss in range(n_surf):
			print('Surface %.2d:' %ss)
			Surf=MS.Surfs[ss]

			# generate non-zero field of external force
			Surf.u_ext[0,:,:]=Surf.u_ext[0,:,:]-20.0
			Surf.u_ext[1,:,:]=Surf.u_ext[1,:,:]+60.0
			Surf.u_ext[2,:,:]=Surf.u_ext[2,:,:]+30.0
			Surf.u_ext=Surf.u_ext+np.random.rand(*Surf.u_ext.shape)

			### analytical derivative
			# ind velocities computed already
			Surf.get_input_velocities_at_collocation_points()
			Der=assembly.uc_dncdzeta(Surf)

			### numerical derivative
			#Surf.get_normal_input_velocities_at_collocation_points()
			u_tot0=Surf.u_ind_coll+Surf.u_input_coll
			u_norm0=Surf.project_coll_to_normal(u_tot0)
			u_norm0_vec=u_norm0.reshape(-1,order='C')
			zeta0=Surf.zeta
			DerNum=np.zeros(Der.shape)

			Steps=np.array([1e-2,1e-3,1e-4,1e-5,1e-6])
			Er_max=0.0*Steps

			for ss in range(len(Steps)):
				step=Steps[ss]
				for jj in range(3*Surf.maps.Kzeta):
					# perturb
					cc_pert=Surf.maps.ind_3d_vert_vect[0][jj]
					mm_pert=Surf.maps.ind_3d_vert_vect[1][jj]
					nn_pert=Surf.maps.ind_3d_vert_vect[2][jj]
					zeta_pert=zeta0.copy()
					zeta_pert[cc_pert,mm_pert,nn_pert]+=step
					# calculate new normal velocity
					Surf_pert=surface.AeroGridSurface(Surf.maps,zeta=zeta_pert,
										   	  u_ext=Surf.u_ext,gamma=Surf.gamma)
					u_norm=Surf_pert.project_coll_to_normal(u_tot0)
					u_norm_vec=u_norm.reshape(-1,order='C')
					# FD derivative
					DerNum[:,jj]=(u_norm_vec-u_norm0_vec)/step

				er_max=np.max(np.abs(Der-DerNum))
				print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
				assert er_max<5e1*step, 'Error larger than 50 times step size'
				Er_max[ss]=er_max

			# assert error decreases with step size
			for ss in range(1,len(Steps)):
				assert Er_max[ss]<Er_max[ss-1],\
				                'Error not decreasing as FD step size is reduced'
			print('------------------------------------------------------------ OK')

			if PlotFlag:
				fig = plt.figure('Spy Der',figsize=(10,4))
				ax1 = fig.add_subplot(121)
				ax1.spy(Der,precision=step)
				ax2 = fig.add_subplot(122)
				ax2.spy(DerNum,precision=step)
				plt.show()



	def test_dfqsdgamma_vrel0(self):


		print('----------------------------- Testing assembly.dfqsdgamma_vrel0')

		MS=self.MS
		n_surf=MS.n_surf

		Der_list,Der_star_list=assembly.dfqsdgamma_vrel0(MS.Surfs,MS.Surfs_star)
		Er_max=[]
		Er_max_star=[]

		Steps=[1e-2,1e-4,1e-6,]


		for ss in range(n_surf):

			Der_an=Der_list[ss]
			Der_star_an=Der_star_list[ss]

			Surf=MS.Surfs[ss]
			Surf_star=MS.Surfs_star[ss]
			M,N=Surf.maps.M,Surf.maps.N
			K=Surf.maps.K

			fqs0=Surf.fqs.copy()
			gamma0=Surf.gamma.copy()

			for step in Steps:
				Der_num=0.0*Der_an
				Der_star_num=0.0*Der_star_an

				### Bound
				for pp in range(K):
					mm=Surf.maps.ind_2d_pan_scal[0][pp]
					nn=Surf.maps.ind_2d_pan_scal[1][pp]	
					Surf.gamma=gamma0.copy()
					Surf.gamma[mm,nn]+=step
					Surf.get_joukovski_qs(gammaw_TE=Surf_star.gamma[0,:])
					df=(Surf.fqs-fqs0)/step
					Der_num[:,pp]=df.reshape(-1,order='C')

				er_max=np.max(np.abs(Der_an-Der_num))
				print('Surface %.2d - bound:' %ss)
				print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
				assert er_max<5e1*step, 'Error larger than 50 times step size'
				Er_max.append(er_max)

				### Wake
				Surf.gamma=gamma0.copy()
				gammaw_TE0=Surf_star.gamma[0,:].copy()
				M_star,N_star=Surf_star.maps.M,Surf_star.maps.N
				K_star=Surf_star.maps.K
				for nn in range(N):
					pp=np.ravel_multi_index( (0,nn), (M_star,N_star))

					gammaw_TE=gammaw_TE0.copy()
					gammaw_TE[nn]+=step
					Surf.get_joukovski_qs(gammaw_TE=gammaw_TE)
					df=(Surf.fqs-fqs0)/step
					Der_star_num[:,pp]=df.reshape(-1,order='C')

				er_max=np.max(np.abs(Der_star_an-Der_star_num))
				print('Surface %.2d - wake:' %ss)
				print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
				assert er_max<5e1*step, 'Error larger than 50 times step size'
				Er_max_star.append(er_max)
			Surf.gamma=gamma0.copy()


			### Warning: this test fails: the dependency on gamma is linear, hence
			# great accuracy is obtained even with large steps. In fact, reducing
			# the step quickly introduced round-off error.

			# # assert error decreases with step size
			# for ii in range(1,len(Steps)):
			# 	assert Er_max[ii]<Er_max[ii-1],\
			# 	                'Error not decreasing as FD step size is reduced'
			# 	assert Er_max_star[ii]<Er_max_star[ii-1],\
			# 	                'Error not decreasing as FD step size is reduced'



	def test_dfqsdzeta_vrel0(self):
		'''
		Note: the get_joukovski_qs method re-computes the induced velocity 
		at the panel segments. A copy of Surf is required to ensure that other
		tests are not affected.
		'''

		print('------------------------------ Testing assembly.dfqsdzeta_vrel0')

		MS=self.MS
		n_surf=MS.n_surf

		Der_list=assembly.dfqsdzeta_vrel0(MS.Surfs,MS.Surfs_star)
		Er_max=[]

		Steps=[1e-2,1e-4,1e-6,]


		for ss in range(n_surf):

			Der_an=Der_list[ss]


			Surf=copy.deepcopy(MS.Surfs[ss])
			#Surf_star=MS.Surfs_star[ss]
			M,N=Surf.maps.M,Surf.maps.N
			K=Surf.maps.K
			Kzeta=Surf.maps.Kzeta

			fqs0=Surf.fqs.copy()
			zeta0=Surf.zeta.copy()

			for step in Steps:
				Der_num=0.0*Der_an

				for kk in range(3*Kzeta):
					Surf.zeta=zeta0.copy()	
					ind_3d=np.unravel_index(kk, (3,M+1,N+1) )								
					Surf.zeta[ind_3d]+=step
					Surf.get_joukovski_qs(gammaw_TE=MS.Surfs_star[ss].gamma[0,:])
					df=(Surf.fqs-fqs0)/step
					Der_num[:,kk]=df.reshape(-1,order='C')

				er_max=np.max(np.abs(Der_an-Der_num))
				print('Surface %.2d - bound:' %ss)
				print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
				assert er_max<5e1*step, 'Error larger than 50 times step size'
				Er_max.append(er_max)

			# fig = plt.figure('Spy Der',figsize=(10,4))
			# ax1 = fig.add_subplot(121)
			# ax1.spy(Der_an,precision=step)
			# ax2 = fig.add_subplot(122)
			# ax2.spy(Der_num,precision=step)
			# plt.show()

			# fig = plt.figure('Spy Error',figsize=(10,4))
			# ax = fig.add_subplot(111)
			# ax.spy( np.abs(Der_an-Der_num),precision=1e2*step)
			# plt.show()

			### Warning: this test fails: the dependency on gamma is linear, hence
			# great accuracy is obtained even with large steps. In fact, reducing
			# the step quickly introduced round-off error.

			# # assert error decreases with step size
			# for ii in range(1,len(Steps)):
			# 	assert Er_max[ii]<Er_max[ii-1],\
			# 	                'Error not decreasing as FD step size is reduced'




	def test_dfqsduinput(self):
		'''
		Step change in input velocity is allocated to both u_ext and zeta_dot
		'''

		print('---------------------------------- Testing assembly.dfqsduinput')

		MS=self.MS
		n_surf=MS.n_surf

		Der_list=assembly.dfqsduinput(MS.Surfs,MS.Surfs_star)
		Er_max=[]

		Steps=[1e-2,1e-4,1e-6,]


		for ss in range(n_surf):

			Der_an=Der_list[ss]


			#Surf=copy.deepcopy(MS.Surfs[ss])
			Surf=MS.Surfs[ss]
			#Surf_star=MS.Surfs_star[ss]
			M,N=Surf.maps.M,Surf.maps.N
			K=Surf.maps.K
			Kzeta=Surf.maps.Kzeta

			fqs0=Surf.fqs.copy()
			u_ext0=Surf.u_ext.copy()
			zeta_dot0=Surf.zeta_dot.copy()

			for step in Steps:
				Der_num=0.0*Der_an

				for kk in range(3*Kzeta):

					Surf.u_ext=u_ext0.copy()
					Surf.zeta_dot=zeta_dot0.copy()

					ind_3d=np.unravel_index(kk, (3,M+1,N+1) )								
					Surf.u_ext[ind_3d]+=0.5*step
					Surf.zeta_dot[ind_3d]+=-0.5*step
					
					Surf.get_input_velocities_at_segments()
					Surf.get_joukovski_qs(gammaw_TE=MS.Surfs_star[ss].gamma[0,:])
					df=(Surf.fqs-fqs0)/step
					Der_num[:,kk]=df.reshape(-1,order='C')

				er_max=np.max(np.abs(Der_an-Der_num))
				print('Surface %.2d - bound:' %ss)
				print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
				assert er_max<5e1*step, 'Error larger than 50 times step size'
				Er_max.append(er_max)




	def test_dfqsdvind_gamma(self):


		print('------------------------------ Testing assembly.dfqsdvind_gamma')

		MS=self.MS
		n_surf=MS.n_surf


		# analytical
		Der_list,Der_star_list=assembly.dfqsdvind_gamma(MS.Surfs,MS.Surfs_star)
		
		# allocate numerical 
		Der_list_num=[]
		Der_star_list_num=[]
		for ii in range(n_surf):
			sub=[]
			sub_star=[]
			for jj in range(n_surf):
				sub.append(0.0*Der_list[ii][jj])
				sub_star.append(0.0*Der_star_list[ii][jj])
			Der_list_num.append(sub)
			Der_star_list_num.append(sub_star)

		# store reference circulation and force
		Gamma0=[]
		Gammaw0=[]
		Fqs0=[]
		for ss in range(n_surf):
			Gamma0.append(MS.Surfs[ss].gamma.copy())
			Gammaw0.append(MS.Surfs_star[ss].gamma.copy())
			Fqs0.append(MS.Surfs[ss].fqs.copy())


		# calculate vis FDs
		#Steps=[1e-2,1e-4,1e-6,]
		Steps=[1e-5,]
		step=Steps[0]

		###### bound
		for ss_in in range(n_surf):
			Surf_in=MS.Surfs[ss_in]

			# perturb
			for pp in range(Surf_in.maps.K):
				mm=Surf_in.maps.ind_2d_pan_scal[0][pp]
				nn=Surf_in.maps.ind_2d_pan_scal[1][pp]	
				Surf_in.gamma=Gamma0[ss_in].copy()
				Surf_in.gamma[mm,nn]+=step

				# recalculate induced velocity everywhere
				MS.get_ind_velocities_at_segments(overwrite=True)
				# restore circulation: (include only induced velocity contrib.)
				Surf_in.gamma=Gamma0[ss_in].copy()

				# estimate derivatives
				for ss_out in range(n_surf):

					Surf_out=MS.Surfs[ss_out]
					fqs0=Fqs0[ss_out].copy()
					Surf_out.get_joukovski_qs(
									 gammaw_TE=MS.Surfs_star[ss_out].gamma[0,:])
					df=(Surf_out.fqs-fqs0)/step
					Der_list_num[ss_out][ss_in][:,pp]=df.reshape(-1,order='C')


		###### wake
		for ss_in in range(n_surf):
			Surf_in=MS.Surfs_star[ss_in]

			# perturb
			for pp in range(Surf_in.maps.K):
				mm=Surf_in.maps.ind_2d_pan_scal[0][pp]
				nn=Surf_in.maps.ind_2d_pan_scal[1][pp]	
				Surf_in.gamma=Gammaw0[ss_in].copy()
				Surf_in.gamma[mm,nn]+=step

				# recalculate induced velocity everywhere
				MS.get_ind_velocities_at_segments(overwrite=True)
				# restore circulation: (include only induced velocity contrib.)
				Surf_in.gamma=Gammaw0[ss_in].copy()

				# estimate derivatives
				for ss_out in range(n_surf):

					Surf_out=MS.Surfs[ss_out]
					fqs0=Fqs0[ss_out].copy()
					Surf_out.get_joukovski_qs(
						gammaw_TE=MS.Surfs_star[ss_out].gamma[0,:]) # <--- gammaw_0 needs to be used here!
					df=(Surf_out.fqs-fqs0)/step
					Der_star_list_num[ss_out][ss_in][:,pp]=df.reshape(-1,order='C')


		### check error
		Er_max=[]
		Er_max_star=[]
		for ss_out in range(n_surf):
			for ss_in in range(n_surf):
				Der_an=Der_list[ss_out][ss_in]
				Der_num=Der_list_num[ss_out][ss_in]
				ErMat=Der_an-Der_num
				ermax=np.max(np.abs(ErMat))
				print('Bound%.2d->Bound%.2d\tFDstep\tError'%(ss_in,ss_out))
				print('\t\t\t%.1e\t%.1e' %(step,ermax))
				assert ermax<50*step, 'Test failed!'

				Der_an=Der_star_list[ss_out][ss_in]
				Der_num=Der_star_list_num[ss_out][ss_in]
				ErMat=Der_an-Der_num
				ermax=np.max(np.abs(ErMat))
				print('Wake%.2d->Bound%.2d\tFDstep\tError'%(ss_in,ss_out))
				print('\t\t\t%.1e\t%.1e' %(step,ermax))
				assert ermax<50*step, 'Test failed!'

				# fig = plt.figure('Spy Der',figsize=(10,4))
				# ax1 = fig.add_subplot(111)
				# ax1.spy(ErMat,precision=50*step)
				# plt.show()



	def test_dvinddzeta(self):
		'''
		For each output surface, there induced velocity is computed, all other
		surfaces are looped. 
		For wakes, only TE is displaced.
		'''

		def comp_vind(zetac,MS):				
			# comute induced velocity
			V=np.zeros((3,))
			for ss in range(n_surf):
				Surf_in=MS.Surfs[ss]
				Surf_star_in=MS.Surfs_star[ss]
				V+=Surf_in.get_induced_velocity(zetac)
				V+=Surf_star_in.get_induced_velocity(zetac)
			return V

		print('----------------------------------- Testing assembly.dvinddzeta')

		MS=self.MS
		n_surf=MS.n_surf
		zetac=.5*(MS.Surfs[0].zeta[:,1,2]+MS.Surfs[0].zeta[:,1,3])

		Dercoll=np.zeros((3,3))
		Dervert_list=[]
		for ss_in in range(n_surf):
			dcoll_b,dvert_b=assembly.dvinddzeta(zetac,MS.Surfs[ss_in],IsBound=True)
			dcoll_w,dvert_w=assembly.dvinddzeta(zetac,MS.Surfs_star[ss_in],
								IsBound=False,M_in_bound=MS.Surfs[ss_in].maps.M)
			Dercoll+=dcoll_b+dcoll_w
			Dervert_list.append(dvert_b+dvert_w)

		# allocate numerical 
		Dercoll_num=np.zeros((3,3))
		Dervert_list_num=[]
		for ii in range(n_surf):
			Dervert_list_num.append(0.0*Dervert_list[ii])
			
		# store reference grid
		Zeta0=[]
		Zeta0_star=[]
		for ss in range(n_surf):
			Zeta0.append(MS.Surfs[ss].zeta.copy())
			Zeta0_star.append(MS.Surfs_star[ss].zeta.copy())
		V0=comp_vind(zetac,MS)

		# calculate vis FDs
		#Steps=[1e-2,1e-4,1e-6,]
		Steps=[1e-6,]
		step=Steps[0]

		### vertices
		for ss_in in range(n_surf):
			Surf_in=MS.Surfs[ss_in]
			Surf_star_in=MS.Surfs_star[ss_in]
			M_in,N_in=Surf_in.maps.M,Surf_in.maps.N

			# perturb
			for kk in range(3*Surf_in.maps.Kzeta):
				cc,mm,nn=np.unravel_index( kk, (3,M_in+1,N_in+1) )

				# perturb bound
				Surf_in.zeta=Zeta0[ss_in].copy()
				Surf_in.zeta[cc,mm,nn]+=step
				# perturb wake TE
				if mm==M_in:
					Surf_star_in.zeta=Zeta0_star[ss_in].copy()
					Surf_star_in.zeta[cc,0,nn]+=step

				# recalculate induced velocity everywhere
				Vnum=comp_vind(zetac,MS)
				dv=(Vnum-V0)/step
				Dervert_list_num[ss_in][:,kk]=dv.reshape(-1,order='C')

				# restore
				Surf_in.zeta=Zeta0[ss_in].copy()
				if mm==M_in:
					Surf_star_in.zeta=Zeta0_star[ss_in].copy()

		### check error at colloc
		Dercoll_num=np.zeros((3,3))
		for cc in range(3):
			zetac_pert=zetac.copy()
			zetac_pert[cc]+=step 
			Vnum=comp_vind(zetac_pert,MS)
			Dercoll_num[:,cc]=(Vnum-V0)/step
		ercoll=np.max(np.abs(Dercoll-Dercoll_num))
		print('Error coll.\tFDstep\tErrAbs')	
		print('\t\t%.1e\t%.1e' %(step,ercoll))	
		assert ercoll<10*step, 'Error at collocation point' 

		### check error at vert
		for ss_in in range(n_surf):
			Der_an=Dervert_list[ss_in]
			Der_num=Dervert_list_num[ss_in]
			ermax,ErAbs,ErRel=max_error_tensor(Der_an,Der_num)

			# max absolute error
			ermax=np.max(ErAbs)
			# relative error at max abs error point
			iimax=np.unravel_index(np.argmax(ErAbs),ErAbs.shape)
			ermax_rel=ErRel[iimax]
			print('Bound and wake%.2d\tFDstep\tErrAbs\tErrRel'%ss_in)
			print('\t\t\t%.1e\t%.1e\t%.1e' %(step,ermax,ermax_rel))
			assert ercoll<10*step, 'Error at vertices'

			fig=plt.figure('Spy Er vs coll derivs',figsize=(12,4))
			ax1=fig.add_subplot(121)
			ax1.spy(ErAbs,precision=1e2*step)
			ax1.set_title('error abs %d' %(ss_in))
			ax2=fig.add_subplot(122)
			ax2.spy(ErRel,precision=1e2*step)
			ax2.set_title('error rel %d' %(ss_in))
			#plt.show()
			plt.close()



	def test_dfqsdvind_zeta(self):
		'''
		For each output surface, there induced velocity is computed, all other
		surfaces are looped. 
		For wakes, only TE is displaced.
		'''

		print('------------------------------- Testing assembly.dfqsdvind_zeta')

		MS=self.MS
		n_surf=MS.n_surf

		# analytical
		Dercoll_list,Dervert_list=assembly.dfqsdvind_zeta(MS.Surfs,MS.Surfs_star)
		
		# allocate numerical 
		Derlist_num=[]
		for ii in range(n_surf):
			sub=[]
			for jj in range(n_surf):
				sub.append(0.0*Dervert_list[ii][jj])
			Derlist_num.append(sub)
			
		# store reference circulation and force
		Zeta0=[]
		Zeta0_star=[]
		Fqs0=[]
		for ss in range(n_surf):
			Zeta0.append(MS.Surfs[ss].zeta.copy())
			Zeta0_star.append(MS.Surfs_star[ss].zeta.copy())
			Fqs0.append(MS.Surfs[ss].fqs.copy())

		# calculate vis FDs
		#Steps=[1e-2,1e-4,1e-6,]
		Steps=[1e-6,]
		step=Steps[0]

		### loop input surfs
		for ss_in in range(n_surf):
			Surf_in=MS.Surfs[ss_in]
			Surf_star_in=MS.Surfs_star[ss_in]
			M_in,N_in=Surf_in.maps.M,Surf_in.maps.N

			# perturb
			for kk in range(3*Surf_in.maps.Kzeta):
				cc,mm,nn=np.unravel_index( kk, (3,M_in+1,N_in+1) )

				# perturb bound
				Surf_in.zeta=Zeta0[ss_in].copy()
				Surf_in.zeta[cc,mm,nn]+=step
				# perturb wake TE
				if mm==M_in:
					Surf_star_in.zeta=Zeta0_star[ss_in].copy()
					Surf_star_in.zeta[cc,0,nn]+=step

				# recalculate induced velocity everywhere
				MS.get_ind_velocities_at_segments(overwrite=True)
				# restore zeta: (include only induced velocity contrib.)
				Surf_in.zeta=Zeta0[ss_in].copy()
				Surf_star_in.zeta=Zeta0_star[ss_in].copy()
				# estimate derivatives
				for ss_out in range(n_surf):

					Surf_out=MS.Surfs[ss_out]
					fqs0=Fqs0[ss_out].copy()
					Surf_out.get_joukovski_qs(
									 gammaw_TE=MS.Surfs_star[ss_out].gamma[0,:])
					df=(Surf_out.fqs-fqs0)/step
					Derlist_num[ss_out][ss_in][:,kk]=df.reshape(-1,order='C')

		### check error
		for ss_out in range(n_surf):
			for ss_in in range(n_surf):
				Der_an=Dervert_list[ss_out][ss_in].copy()
				if ss_in==ss_out:
					Der_an=Der_an+Dercoll_list[ss_out]
				Der_num=Derlist_num[ss_out][ss_in]
				ermax, ErAbs, ErRel=max_error_tensor(Der_an,Der_num)

				# max absolute error
				ermax=np.max(ErAbs)
				# relative error at max abs error point
				iimax=np.unravel_index(np.argmax(ErAbs),ErAbs.shape)
				ermax_rel=ErRel[iimax]


				print('Bound%.2d->Bound%.2d\tFDstep\tErrAbs\tErrRel'%(ss_in,ss_out))
				print('\t\t\t%.1e\t%.1e\t%.1e' %(step,ermax,ermax_rel))
				assert ermax<5e2*step and ermax_rel<50*step, 'Test failed!'

				fig=plt.figure('Spy Er vs coll derivs',figsize=(12,4))

				ax1=fig.add_subplot(131)
				ax1.spy(ErAbs,precision=1e2*step)
				ax1.set_title('error abs %d to %d' %(ss_in,ss_out))

				ax2=fig.add_subplot(132)
				ax2.spy(ErRel,precision=1e2*step)
				ax2.set_title('error rel %d to %d' %(ss_in,ss_out))

				ax3=fig.add_subplot(133)
				ax3.spy(Dercoll_list[ss_out],precision=50*step)
				ax3.set_title('Dcoll an. %d to %d' %(ss_out,ss_out))		
				#plt.show()
				plt.close()





if __name__=='__main__':

	#unittest.main()
	T=Test_assembly()
	T.setUp()

	# ### force equation (qs term)
	T.test_dvinddzeta()
	T.test_dfqsdvind_zeta()
	T.setUp()
	T.test_dfqsdvind_gamma()
	T.test_dfqsduinput()
	T.test_dfqsdzeta_vrel0()
	T.test_dfqsdgamma_vrel0()

	### state equation terms
	T.test_uc_dncdzeta()
	T.test_nc_dqcdzeta_bound_to_bound()
	T.test_nc_dqcdzeta_wake_to_bound()

	









