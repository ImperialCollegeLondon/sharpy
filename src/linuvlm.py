'''
Linearise UVLM solver
S. Maraniello, 7 Jun 2018
'''

import numpy as np
import scipy.linalg as scalg
from IPython import embed
import time

import interp
import multisurfaces
import assembly as ass # :D


class Static():


	def __init__(self,tsdata):
		
		print('Initialising Static solver class...')
		t0=time.time()

		MS=multisurfaces.MultiAeroGridSurfaces(tsdata)
		MS.get_ind_velocities_at_collocation_points()
		MS.get_input_velocities_at_collocation_points()
		MS.get_ind_velocities_at_segments()
		MS.get_input_velocities_at_segments()

		# define total sizes
		self.K=sum(MS.KK)
		self.K_star=sum(MS.KK_star)
		self.Kzeta=sum(MS.KKzeta)
		self.Kzeta_star=sum(MS.KKzeta_star)		
		self.MS=MS

		# define input perturbation
		self.zeta=np.zeros((3*self.Kzeta))
		self.zeta_dot=np.zeros((3*self.Kzeta))
		self.u_ext=np.zeros((3*self.Kzeta))

		# profiling output
		self.prof_out='./asbly.prof'

		self.time_init=time.time()-t0
		print('Initialisation done in %.2f sec' %self.time_init)




	def assemble_profiling(self):
		import cProfile
		cProfile.runctx('self.assemble()',globals(),locals(),filename=self.prof_out)


	def assemble(self):
		'''
		Assemble global matrices
		'''
		print('Assembly started...')
		MS=self.MS
		t0=time.time()

		# ----------------------------------------------------------- state eq.
		List_uc_dncdzeta=ass.uc_dncdzeta(MS.Surfs)
		List_nc_dqcdzeta_coll,List_nc_dqcdzeta_vert=\
										 ass.nc_dqcdzeta(MS.Surfs,MS.Surfs_star)
		List_AICs,List_AICs_star=ass.AICs(MS.Surfs,MS.Surfs_star,
											  target='collocation',Project=True)
		List_Wnv=[]
		for ss in range(MS.n_surf):
			List_Wnv.append(
				interp.get_Wnv_vector(MS.Surfs[ss],
											   MS.Surfs[ss].aM,MS.Surfs[ss].aN))

		# zeta derivatives
		self.Ducdzeta=np.block(List_nc_dqcdzeta_vert)
		del List_nc_dqcdzeta_vert
		self.Ducdzeta+=scalg.block_diag(*List_nc_dqcdzeta_coll)
		del List_nc_dqcdzeta_coll
		self.Ducdzeta+=scalg.block_diag(*List_uc_dncdzeta)
		del List_uc_dncdzeta
		# assemble input velocity derivatives
		self.Ducdu_ext=scalg.block_diag(*List_Wnv)
		del List_Wnv

		### assemble global matrices
		# Gamma derivatives (global AICs) # <--- keep for dynamic
		# AIC=np.block(List_AICs)
		#AIC_star=np.block(List_AICs_star) 

		### Condense Gammaw terms
		for ss_out in range(MS.n_surf):
			K=MS.KK[ss_out]
			for ss_in in range(MS.n_surf):
				N_star=MS.NN_star[ss_in]
				aic=List_AICs[ss_out][ss_in] 	  # bound
				aic_star=List_AICs_star[ss_out][ss_in] # wake

				# fold aic_star: sum along chord at each span-coordinate
				aic_star_fold=np.zeros((K,N_star))
				for jj in range(N_star):
					aic_star_fold[:,jj]+=np.sum(aic_star[:,jj::N_star],axis=1)
				aic[:,-N_star:]+=aic_star_fold

		self.AIC=np.block(List_AICs)


		# ---------------------------------------------------------- output eq.

		### Zeta derivatives
		# ... at constant relative velocity		
		self.Dfqsdzeta=scalg.block_diag(
								   *ass.dfqsdzeta_vrel0(MS.Surfs,MS.Surfs_star))
		# ... induced velocity contrib.
		List_coll,List_vert=ass.dfqsdvind_zeta(MS.Surfs,MS.Surfs_star)
		for ss in range(MS.n_surf):
			List_vert[ss][ss]+=List_coll[ss]
		self.Dfqsdzeta+=np.block(List_vert)
		del List_vert, List_coll


		### Input velocities
		self.Dfqsdu_ext=scalg.block_diag(
									   *ass.dfqsduinput(MS.Surfs,MS.Surfs_star))


		### Gamma derivatives
		# ... at constant relative velocity
		List_dfqsdgamma_vrel0,List_dfqsdgamma_star_vrel0=\
							   		ass.dfqsdgamma_vrel0(MS.Surfs,MS.Surfs_star)
		self.Dfqsdgamma=scalg.block_diag(*List_dfqsdgamma_vrel0)
		self.Dfqsdgamma_star=scalg.block_diag(*List_dfqsdgamma_star_vrel0)
		del List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0
		# ... induced velocity contrib.
		List_dfqsdvind_gamma,List_dfqsdvind_gamma_star=\
									 ass.dfqsdvind_gamma(MS.Surfs,MS.Surfs_star)
		self.Dfqsdgamma+=np.block(List_dfqsdvind_gamma)
		self.Dfqsdgamma_star+=np.block(List_dfqsdvind_gamma_star)
		del List_dfqsdvind_gamma, List_dfqsdvind_gamma_star


		### Condense Gammaw terms
		# not required for output
		self.time_asbly=time.time()-t0
		print('Assembly done in %.2f sec' %self.time_asbly)


	def solve(self):
		'''
		solve for bound Gamma
		'''

		MS=self.MS
		t0=time.time()

		### state
		bv=np.dot(self.Ducdu_ext,self.u_ext-self.zeta_dot )+\
												 np.dot(self.Ducdzeta,self.zeta)
		self.gamma=np.linalg.solve(self.AIC,-bv)

		### retrieve gamma over wake
		gamma_star=[]
		Ktot=0
		for ss in range(MS.n_surf):
			Ktot+=MS.Surfs[ss].maps.K
			N=MS.Surfs[ss].maps.N
			Mstar=MS.Surfs_star[ss].maps.M
			gamma_star.append(np.concatenate( Mstar*[self.gamma[Ktot-N:Ktot]] ))
		gamma_star=np.concatenate(gamma_star)

		### compute steady force
		self.fqs=np.dot(self.Dfqsdgamma,self.gamma) +\
					np.dot(self.Dfqsdgamma_star,gamma_star) +\
						np.dot(self.Dfqsdzeta,self.zeta) +\
								np.dot(self.Dfqsdu_ext,self.u_ext-self.zeta_dot)

		self.time_sol=time.time()-t0
		print('Solution done in %.2f sec' %self.time_sol)


	def reshape(self):
		'''Reshapes state/output according to sharpy format'''


		MS=self.MS
		if not hasattr(self,'gamma') or not hasattr(self,'fqs'):
			raise NameError('State and output not found')
		
		self.Gamma=[]
		self.Fqs=[]

		Ktot,Kzeta_tot=0,0
		for ss in range(MS.n_surf):

			M,N=MS.Surfs[ss].maps.M,MS.Surfs[ss].maps.N
			K,Kzeta=MS.Surfs[ss].maps.K,MS.Surfs[ss].maps.Kzeta

			iivec=range(Ktot,Ktot+K)
			self.Gamma.append( self.gamma[iivec].reshape((M,N)) )
			
			iivec=range(Kzeta_tot,Kzeta_tot+3*Kzeta)
			self.Fqs.append( self.fqs[iivec].reshape((3,M+1,N+1)) )

			Ktot+=K
			Kzeta_tot+=3*Kzeta


	def total_forces(self):

		if not hasattr(self,'Gamma') or not hasattr(self,'Fqs'):		
			self.reshape()

		self.Ftot=np.zeros((3,))
		for ss in range(self.MS.n_surf):
			for cc in range(3):
				self.Ftot[cc]+=np.sum(self.Fqs[ss][cc,:,:])



if __name__=='__main__':

	import timeit
	import read
	import matplotlib.pyplot as plt 

	# # # select test case
	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
	haero=read.h5file(fname)
	tsdata=haero.ts00000
	Sol=Static(tsdata)


	Sol.assemble_profiling()
	embed()

	Sol.assemble()
	# Solve:
	Sol.u_ext=np.ones((3*Sol.Kzeta,))
	Sol.solve()
	Sol.reshape()
	Sol.total_forces()
	print(Sol.Ftot)


	# fname='../test/h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
	# haero=read.h5file(fname)
	# tsdata=haero.ts00000
	# Sol2=Static(tsdata)
	# Sol2.assemble()
	# #timeit.timeit('Sol2.assemble()')
	# Sol2.u_ext=np.ones((3*Sol2.Kzeta,))
	# Sol2.solve()
	# Sol2.reshape()
	# Sol2.total_forces()
	# print(Sol2.Ftot)

