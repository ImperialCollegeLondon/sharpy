'''
Linearise UVLM solver
S. Maraniello, 7 Jun 2018
'''

import numpy as np
import scipy.linalg as scalg
from IPython import embed

import interp
import multisurfaces
import assembly as ass # :D


class Static():




	def __init__(self,tsdata):
				


		MS=multisurfaces.MultiAeroGridSurfaces(tsdata)
		MS.get_ind_velocities_at_collocation_points()
		MS.get_input_velocities_at_collocation_points()

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



	def assemble(self):
		'''
		Assemble global matrices
		'''
		MS=self.MS

		# get matrices
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

		## zeta derivatives
		self.Ducdzeta=np.block(List_nc_dqcdzeta_vert)
		self.Ducdzeta+=scalg.block_diag(*List_nc_dqcdzeta_coll)
		self.Ducdzeta+=scalg.block_diag(*List_uc_dncdzeta)
		# assemble input velocity derivatives
		self.Ducdu_ext=scalg.block_diag(*List_Wnv)

		### assemble global matrices
		# Gamma derivatives (global AICs) # <--- keep for dynamic
		# AIC=np.block(List_AICs)
		#AIC_star=np.block(List_AICs_star) 

		### Condense Gammw terms
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


	def solve(self):
		'''
		solve for bound Gamma
		'''

		bv=np.dot(self.Ducdu_ext,self.u_ext-self.zeta_dot )+\
												 np.dot(self.Ducdzeta,self.zeta)
		self.gamma=np.linalg.solve(self.AIC,-bv)




if __name__=='__main__':

	import read
	import matplotlib.pyplot as plt 

	# select test case
	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
	fname='../test/h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
	haero=read.h5file(fname)
	tsdata=haero.ts00000

	Sol=Static(tsdata)
	MS=Sol.MS
	Sol.assemble()

	# Solve:
	Sol.u_ext=np.random.rand(3*Sol.Kzeta)
	Sol.solve()

