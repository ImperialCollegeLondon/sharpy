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

sys.path.append(os.path.abspath('/home/arturo/code/sharpy/sharpy/utils'))
import assembly, multisurfaces, surface, libuvlm, h5utils

from IPython import embed
np.set_printoptions(linewidth=200,precision=3)

omega = np.array([1.6,0.0,0.0])

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

	nnzvec=Pder_an!=0
	Erel=np.zeros(Pder_an.shape)
	Erel[nnzvec]=np.abs(Eabs[nnzvec]/Pder_an[nnzvec])

	# Relative error check: remove NaN and inf...
	iifinite=np.isfinite(Erel)
	err_max=0.0
	for err_here in Erel[iifinite]:
		if np.abs(err_here)>err_max:
			err_max=err_here

	# Zero elements check
	iizero=np.abs(Pder_an)<1e-15
	for der_here in Pder_num[iizero]:
		if np.abs(der_here)>err_max:
			err_max=der_here

	return err_max, Eabs, Erel




class Test_assembly(unittest.TestCase):
	''' Test methods into assembly module '''

	def setUp(self):

		# select test case
		#fname='./h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
		#fname='./h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
		fname='./basic_rotating_wing/basic_wing.data.h5'
		haero=h5utils.readh5(fname)
		tsdata=haero.data.aero.timestep_info[0]

		MS=multisurfaces.MultiAeroGridSurfaces(tsdata, omega=omega)
		MS.get_normal_ind_velocities_at_collocation_points()
		MS.verify_non_penetration()
		MS.verify_aic_coll()
		MS.get_joukovski_qs()
		MS.verify_joukovski_qs()
		self.MS=MS



	def test_nc_dqcdzeta(self):
		'''
		For each output surface, where induced velocity is computed, all other
		surfaces are looped.
		For wakes, only TE is displaced.
		'''

		print('----------------------------- Testing assembly.test_nc_dqcdzeta')

		MS=self.MS
		n_surf=MS.n_surf

		# analytical
		Dercoll_list,Dervert_list=assembly.nc_dqcdzeta(MS.Surfs,MS.Surfs_star)

		# allocate numerical
		Derlist_num=[]
		for ii in range(n_surf):
			sub=[]
			for jj in range(n_surf):
				sub.append(0.0*Dervert_list[ii][jj])
			Derlist_num.append(sub)

		# store reference circulation and normal induced velocities
		MS.get_normal_ind_velocities_at_collocation_points()
		Zeta0=[]
		Zeta0_star=[]
		Vind0=[]
		N0=[]
		ZetaC0=[]
		for ss in range(n_surf):
			Zeta0.append(MS.Surfs[ss].zeta.copy())
			ZetaC0.append(MS.Surfs[ss].zetac.copy('F'))
			Zeta0_star.append(MS.Surfs_star[ss].zeta.copy())
			Vind0.append(MS.Surfs[ss].u_ind_coll_norm.copy())
			N0.append(MS.Surfs[ss].normals.copy())

		# calculate vis FDs
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

				# perturb bound. vertices and collocation
				Surf_in.zeta=Zeta0[ss_in].copy()
				Surf_in.zeta[cc,mm,nn]+=step
				Surf_in.generate_collocations()

				# perturb wake TE
				if mm==M_in:
					Surf_star_in.zeta=Zeta0_star[ss_in].copy()
					Surf_star_in.zeta[cc,0,nn]+=step

				### prepare output surfaces
				# - ensure normals are unchanged
				# - del ind. vel on output to ensure they are re-computed
				for ss_out in range(n_surf):
					Surf_out=MS.Surfs[ss_out]
					Surf_out.normals=N0[ss_out].copy()
					del Surf_out.u_ind_coll_norm
					try:
						del Surf_out.u_ind_coll
					except AttributeError:
						pass

				### recalculate
				MS.get_normal_ind_velocities_at_collocation_points()

				# restore
				Surf_in.zeta=Zeta0[ss_in].copy()
				Surf_in.zetac=ZetaC0[ss_in].copy('F')
				Surf_star_in.zeta=Zeta0_star[ss_in].copy()

				# estimate derivatives
				for ss_out in range(n_surf):
					Surf_out=MS.Surfs[ss_out]
					dvind=(Surf_out.u_ind_coll_norm-Vind0[ss_out])/step
					Derlist_num[ss_out][ss_in][:,kk]=dvind.reshape(-1,order='C')


		### check error
		for ss_out in range(n_surf):
			for ss_in in range(n_surf):
				Der_an=Dervert_list[ss_out][ss_in].copy()
				if ss_in==ss_out:
					Der_an=Der_an+Dercoll_list[ss_out]
				Der_num=Derlist_num[ss_out][ss_in]
				_,ErAbs,ErRel=max_error_tensor(Der_an,Der_num)

				# max absolute error
				ermax=np.max(ErAbs)
				# relative error at max abs error point
				iimax=np.unravel_index(np.argmax(ErAbs),ErAbs.shape)
				ermax_rel=ErRel[iimax]

				print('Bound%.2d->Bound%.2d\tFDstep\tErrAbs\tErrRel'%(ss_in,ss_out))
				print('\t\t\t%.1e\t%.1e\t%.1e' %(step,ermax,ermax_rel))
				#assert ermax<50*step and ermax_rel<50*step, embed()#'Test failed!'

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

    ########### ams start #########

	def test_dfqsdzeta_omega(self):
		'''
		Note: the get_joukovski_qs method re-computes the induced velocity
		at the panel segments. A copy of Surf is required to ensure that other
		tests are not affected.
		'''

		print('------------------------------ Testing assembly.dfqsdzeta_omega')

		# rename
		MS=self.MS
		n_surf=MS.n_surf

		# Compute the anaytical derivative of the case
		Der_an_list=assembly.dfqsdzeta_omega(MS.Surfs,MS.Surfs_star)

		# Initialize
		Er_max=[]

		# Define steps to run
		Steps=[1e-2,1e-4,1e-6,]

		for ss in range(n_surf):
			# Select the surface with the analytica derivatives
			Der_an=Der_an_list[ss]

			# Copy to avoid modifying the original for other tests
			Surf=copy.deepcopy(MS.Surfs[ss])
			# Define variables
			M,N=Surf.maps.M,Surf.maps.N
			K=Surf.maps.K
			Kzeta=Surf.maps.Kzeta

			# Save the reference values at equilibrium
			fqs0=Surf.fqs.copy()
			zeta0=Surf.zeta.copy()
			u_input_seg0=Surf.u_input_seg.copy()

			for step in Steps:
				# Initialize
				Der_num = 0.0*Der_an

				# Loop through the different grid modifications (three directions per vertex point)
				for kk in range(3*Kzeta):
					# Initialize to remove previous movements
					Surf.zeta=zeta0.copy()
					# Define DoFs where modifications will take place and modify the grid
					ind_3d=np.unravel_index(kk, (3,M+1,N+1) )
					Surf.zeta[ind_3d]+=step
					# Recompute get_ind_velocities_at_segments and recover the previous grid
					Surf.get_input_velocities_at_segments()
					Surf.zeta=zeta0.copy()
					# Compute new forces
					Surf.get_joukovski_qs(gammaw_TE=MS.Surfs_star[ss].gamma[0,:])
					df=(Surf.fqs-fqs0)/step
					Der_num[:,kk]=df.reshape(-1,order='C')

				er_max=np.max(np.abs(Der_an-Der_num))
				print('Surface %.2d - bound:' %ss)
				print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
				assert er_max<5e1*step, 'Error larger than 50 times step size'
				Er_max.append(er_max)

    ########### ams end #########

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
		#if ercoll>10*step: embed()
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



	def test_dfunstdgamma_dot(self):
		'''
		Test derivative of unsteady aerodynamic force with respect to changes in
		panel circulation.

		Warning: test assumes the derivative of the unsteady force only depends on
		Gamma_dot, which is true only for steady-state linearisation points
		'''

		MS=self.MS
		Ders_an=assembly.dfunstdgamma_dot(MS.Surfs)


		step=1e-6
		Ders_num=[]
		n_surf=len(MS.Surfs)
		for ss in range(n_surf):

			Surf=MS.Surfs[ss]
			Kzeta,K=Surf.maps.Kzeta,Surf.maps.K
			M,N=Surf.maps.M,Surf.maps.N

			Dnum=np.zeros((3*Kzeta,K))

			# get refernce values
			Surf.get_joukovski_unsteady()
			Gamma_dot0=Surf.gamma_dot.copy()
			F0=Surf.funst.copy()

			for pp in range(K):
				mm,nn=np.unravel_index( pp, (M,N) )

				Surf.gamma_dot=Gamma_dot0.copy()
				Surf.gamma_dot[mm,nn]+=step
				Surf.get_joukovski_unsteady()

				dF=(Surf.funst-F0)/step
				Dnum[:,pp]=dF.reshape(-1)

			# restore
			Surf.gamma_dot=Gamma_dot0.copy()

			### verify
			ermax, ErAbs, ErRel=max_error_tensor(Ders_an[ss],Dnum)

			# max absolute error
			ermax=np.max(ErAbs)
			# relative error at max abs error point
			iimax=np.unravel_index(np.argmax(ErAbs),ErAbs.shape)
			ermax_rel=ErRel[iimax]

			print('Bound%.2d\t\t\tFDstep\tErrAbs\tErrRel'%(ss,))
			print('\t\t\t%.1e\t%.1e\t%.1e' %(step,ermax,ermax_rel))
			assert ermax<5e2*step and ermax_rel<50*step, 'Test failed!'



	def test_wake_prop(self):

		MS=self.MS
		C_list,Cstar_list=assembly.wake_prop(MS.Surfs,MS.Surfs_star)

		n_surf=len(MS.Surfs)
		for ss in range(n_surf):

			Surf=MS.Surfs[ss]
			Surf_star=MS.Surfs_star[ss]
			N=Surf.maps.N
			K_star=Surf_star.maps.K
			C=C_list[ss]
			Cstar=Cstar_list[ss]


			# add noise to circulations
			gamma=Surf.gamma+np.random.rand( *Surf.gamma.shape )
			gamma_star=Surf_star.gamma+np.random.rand( *Surf_star.gamma.shape )


			gvec=np.dot(C,gamma.reshape(-1))+np.dot(Cstar,gamma_star.reshape(-1))

			gvec_ref=np.concatenate((gamma[-1,:],gamma_star[:-1,:].reshape(-1)))

			assert np.max(np.abs(gvec-gvec_ref))<1e-15,\
										  'Prop. from trailing edge not correct'





if __name__=='__main__':

	unittest.main()

	# T=Test_assembly()
	# T.setUp()

	# ### force equation (qs term)
	# T.test_dvinddzeta()
	# T.test_dfqsdvind_zeta() # run setUp after this test

	# T.setUp()
	# T.test_dfqsdvind_gamma()
	# T.test_dfqsduinput()
	# T.test_dfqsdzeta_vrel0()
	# T.test_dfqsdgamma_vrel0()

	# ### state equation terms
	# T.test_uc_dncdzeta()
	# T.test_nc_dqcdzeta()


	### force equation (unsteady)
	# T.test_dfunstdgamma_dot()
