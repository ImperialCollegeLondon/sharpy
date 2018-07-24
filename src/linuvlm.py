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
	'''	Static linear solver '''


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

		self.time_init_sta=time.time()-t0
		print('\t\t\t...done in %.2f sec' %self.time_init_sta)


	def assemble_profiling(self):
		import cProfile
		cProfile.runctx('self.assemble()',globals(),locals(),filename=self.prof_out)


	def assemble(self):
		'''
		Assemble global matrices
		'''
		print('Assembly of static linear equations started...')
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
		print('\t\t\t...done in %.2f sec' %self.time_asbly)


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





class Dynamic(Static):


	def __init__(self,tsdata,dt,integr_order=2,Uref=1.0):

		super().__init__(tsdata)

		self.dt=dt
		self.integr_order=integr_order
		self.Uref=1.0 # ref. velocity for scaling


		if self.integr_order==1:
			Nx=2*self.K+self.K_star
		if self.integr_order==2:
			Nx=3*self.K+self.K_star
			b0,bm1,bp1=-2.,0.5,1.5
		Ny=3*self.Kzeta
		Nu=3*Ny
		self.Nx=Nx
		self.Nu=Nu
		self.Ny=Ny


		# ### rename static methods
		# self.assemble_static=self.assemble
		# self.assemble_static_profiling=self.assemble_profiling
		# self.solve_static=self.solve 
		# self.total_forces_static=self.total_forces

		# print('Initialising Dynamic solver class...')
		# t0=time.time()
		# self.time_init_dyn=time.time()-t0
		# print('\t\t\t...done in %.2f sec' %self.time_init_dyn)


	def nondimvars(self):
		pass

	def dimvars(self):
		pass

	def assemble_ss(self):
		'''
		Produces state-space model
			x_{n+1} = A x_n + B u_{n+1}
			y = C x + D u
		Warning: all matrices are allocated as full!
		'''

		print('State-space realisation started...')
		t0=time.time()	
		MS=self.MS	
		K,K_star=self.K,self.K_star
		Kzeta=self.Kzeta


		# ------------------------------------------------------ determine size
		
		Nx=self.Nx
		Nu=self.Nu
		Ny=self.Ny
		if self.integr_order==2:
			b0,bm1,bp1=-2.,0.5,1.5


		# ----------------------------------------------------------- state eq.
	
		### state terms (A matrix)

		# Aero influence coeffs
		List_AICs,List_AICs_star=ass.AICs(MS.Surfs,MS.Surfs_star,
											  target='collocation',Project=True)
		A0=np.block(List_AICs)
		A0W=np.block(List_AICs_star)
		del List_AICs, List_AICs_star
		LU,P=scalg.lu_factor(A0)
		AinvAW=scalg.lu_solve( (LU,P), A0W)
		del A0, A0W

		# propagation of circ
		List_C,List_Cstar=ass.wake_prop(MS.Surfs,MS.Surfs_star)
		Cgamma=scalg.block_diag(*List_C)
		CgammaW=scalg.block_diag(*List_Cstar)
		del List_C, List_Cstar

		# A matrix assembly
		Ass=np.zeros((Nx,Nx))
		Ass[:K,:K]=-np.dot(AinvAW,Cgamma)
		Ass[:K,K:K+K_star]=-np.dot(AinvAW,CgammaW)
		Ass[K:K+K_star,:K]=Cgamma
		Ass[K:K+K_star,K:K+K_star]=CgammaW

		if self.integr_order==1:
			# delta eq.
			Ass[K+K_star:2*K+K_star,:K]=Ass[:K,:K]-np.eye(K)
			Ass[K+K_star:2*K+K_star,K:K+K_star]=Ass[:K,K:K+K_star]
		if self.integr_order==2:
			# delta eq.
			Ass[K+K_star:2*K+K_star,:K]=bp1*Ass[:K,:K]+b0*np.eye(K)
			Ass[K+K_star:2*K+K_star,K:K+K_star]=bp1*Ass[:K,K:K+K_star]
			Ass[K+K_star:2*K+K_star,K+K_star:2*K+K_star]=0.0
			Ass[K+K_star:2*K+K_star,2*K+K_star:3*K+K_star]=bm1*np.eye(K)
			# identity eq.
			Ass[2*K+K_star:3*K+K_star,:K]=np.eye(K)


		### input terms (B matrix)

		# zeta derivs
		List_uc_dncdzeta=ass.uc_dncdzeta(MS.Surfs)
		List_nc_dqcdzeta_coll,List_nc_dqcdzeta_vert=\
										 ass.nc_dqcdzeta(MS.Surfs,MS.Surfs_star)
		Ducdzeta=np.block(List_nc_dqcdzeta_vert)
		del List_nc_dqcdzeta_vert
		Ducdzeta+=scalg.block_diag(*List_nc_dqcdzeta_coll)
		del List_nc_dqcdzeta_coll
		Ducdzeta+=scalg.block_diag(*List_uc_dncdzeta)
		del List_uc_dncdzeta

		# ext velocity derivs (Wnv0)
		List_Wnv=[]
		for ss in range(MS.n_surf):
			List_Wnv.append(
				interp.get_Wnv_vector(MS.Surfs[ss],
											   MS.Surfs[ss].aM,MS.Surfs[ss].aN))
		#Ducdu_ext=scalg.block_diag(*List_Wnv)
		#AinvWnv0=scalg.lu_solve((LU,P),Ducdu_ext)
		AinvWnv0=scalg.lu_solve( (LU,P), scalg.block_diag(*List_Wnv))
		del List_Wnv

		# B matrix assembly
		Bss=np.zeros((Nx,Nu))
		Bss[:K,:3*Kzeta]=-scalg.lu_solve( (LU,P), Ducdzeta )
		Bss[:K,3*Kzeta:6*Kzeta]= AinvWnv0
		Bss[:K,6*Kzeta:9*Kzeta]=-AinvWnv0
		if self.integr_order==1:
			Bss[K+K_star:2*K+K_star,:]=Bss[:K,:]
		if self.integr_order==2:
			Bss[K+K_star:2*K+K_star,:]=bp1*Bss[:K,:]



		# ---------------------------------------------------------- output eq.

		### state terms (C matrix)
		
		# gamma (at constant relative velocity)
		List_dfqsdgamma_vrel0,List_dfqsdgamma_star_vrel0=\
							   		ass.dfqsdgamma_vrel0(MS.Surfs,MS.Surfs_star)
		Dfqsdgamma=scalg.block_diag(*List_dfqsdgamma_vrel0)
		Dfqsdgamma_star=scalg.block_diag(*List_dfqsdgamma_star_vrel0)
		del List_dfqsdgamma_vrel0, List_dfqsdgamma_star_vrel0
		# gamma (induced velocity contrib.)
		List_dfqsdvind_gamma,List_dfqsdvind_gamma_star=\
									 ass.dfqsdvind_gamma(MS.Surfs,MS.Surfs_star)
		Dfqsdgamma+=np.block(List_dfqsdvind_gamma)
		Dfqsdgamma_star+=np.block(List_dfqsdvind_gamma_star)
		del List_dfqsdvind_gamma, List_dfqsdvind_gamma_star

		# gamma_dot
		Dfunstdgamma_dot=scalg.block_diag( *ass.dfunstdgamma_dot(MS.Surfs) )

		# C matrix assembly
		Css=np.zeros((Ny,Nx))
		Css[:,:K]=Dfqsdgamma
		Css[:,K:K+K_star]=Dfqsdgamma_star
		Css[:,K+K_star:2*K+K_star]=Dfunstdgamma_dot/self.dt



		### input terms (D matrix)
		Dss=np.zeros((Ny,Nu))

		# zeta (at constant relative velocity)
		Dss[:,:3*Kzeta]=scalg.block_diag(*ass.dfqsdzeta_vrel0(MS.Surfs,MS.Surfs_star))
		# zeta (induced velocity contrib)
		List_coll,List_vert=ass.dfqsdvind_zeta(MS.Surfs,MS.Surfs_star)
		for ss in range(MS.n_surf):
			List_vert[ss][ss]+=List_coll[ss]
		Dss[:,:3*Kzeta]+=np.block(List_vert)
		del List_vert, List_coll

		# input velocities
		Dss[:,6*Kzeta:9*Kzeta]=scalg.block_diag(*ass.dfqsduinput(MS.Surfs,MS.Surfs_star) )
		Dss[:,3*Kzeta:6*Kzeta]=-Dss[:,6*Kzeta:9*Kzeta]
	
		self.Ass=Ass 
		self.Bss=Bss
		self.Css=Css
		self.Dss=Dss

		self.time_ss=time.time()-t0
		print('\t\t\t...done in %.2f sec' %self.time_ss)



	def solve_steady(self,usta,method='direct'):
		'''
		Steady state solution from state-space model
		Warning: this method is less efficient than the solver in Static class
		and should be used only for verification purposes.
		'''

		if method=='direct':
			Ass_steady=np.eye(*self.Ass.shape)-self.Ass
			xsta=np.linalg.solve( Ass_steady, np.dot(self.Bss,usta) )
			ysta=np.dot(self.Css,xsta)+np.dot(self.Dss,usta)	

		if method=='recursive':
			tol,er=1e-4,1.0
			Ftot0=np.zeros((3,))
			nn=0
			xsta=np.zeros((self.Nx))
			while er>tol and nn<1000:
				xsta=np.dot( Dyn.Ass,xsta )+np.dot(Dyn.Bss,usta)
				ysta=np.dot(self.Css,xsta)+np.dot(self.Dss,usta)
				Ftot=np.array(
						[ np.sum(ysta[cc*self.Kzeta:(cc+1)*self.Kzeta]) 
															for cc in range(3)])
				er=np.linalg.norm(Ftot-Ftot0)
				Ftot0=Ftot.copy()
				nn+=1
			if er<tol:
				pass # print('Recursive solution found in %.3d iterations'%nn)
			else:
				print('Solution not found! Max. iterations reached with error: %.3e'%er)

		if method=='subsystem':
			Nxsub=self.K+self.K_star
			Asub_steady=np.eye(Nxsub)-self.Ass[:Nxsub,:Nxsub]
			xsub=np.linalg.solve( Asub_steady,np.dot(self.Bss[:Nxsub,:],usta))
			if self.integr_order==1:
				xsta=np.concatenate( (xsub,np.zeros((  self.K,))) ) 
			if self.integr_order==2:
				xsta=np.concatenate( (xsub,np.zeros((2*self.K,))) ) 
			ysta=np.dot(self.Css,xsta)+np.dot(self.Dss,usta)	

		return xsta,ysta




if __name__=='__main__':

	import timeit
	import read
	import matplotlib.pyplot as plt 

	# # # select test case
	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
	haero=read.h5file(fname)
	tsdata=haero.ts00000

	# Static solver
	Sta=Static(tsdata)
	Sta.assemble_profiling()
	Sta.assemble()

	# random input
	Sta.u_ext   =1.0+0.30*np.random.rand(3*Sta.Kzeta)
	Sta.zeta_dot=0.2+0.10*np.random.rand(3*Sta.Kzeta)
	Sta.zeta    =    0.05*(np.random.rand(3*Sta.Kzeta)-1.0)

	Sta.solve()
	Sta.reshape()
	Sta.total_forces()
	print(Sta.Ftot)

	# Dynamic solver
	Dyn=Dynamic(tsdata,dt=0.05,integr_order=2,Uref=1.0)
	Dyn.assemble_ss()


	### Verify dynamic solver

	# steady state solution
	usta=np.concatenate( (Sta.zeta,Sta.zeta_dot,Sta.u_ext) )
	xsta,ysta=Dyn.solve_steady(usta,method='direct')
	xrec,yrec=Dyn.solve_steady(usta,method='recursive')
	xsub,ysub=Dyn.solve_steady(usta,method='subsystem')

	# assert all solutions are matching
	assert max(np.linalg.norm(xsta-xrec), np.linalg.norm(ysta-yrec)),\
								  'Direct and recursive solutions not matching!'

	assert max(np.linalg.norm(xsta-xsub), np.linalg.norm(ysta-ysub)),\
								  'Direct and sub-system solutions not matching!'


	# compare against Static solver solution
	er=np.max(np.abs(ysta-Sta.fqs)/np.linalg.norm(Sta.Ftot)  )
	print('Error force distribution: %.3e' %er)
	assert er<1e-12,'Steady-state force not matching!'

	er=np.max(np.abs( xsta[:Dyn.K]-Sta.gamma ))
	print('Error bound circulation: %.3e' %er)
	assert er<1e-13,'Steady-state gamma not matching!'

	gammaw_ref=np.zeros((Dyn.K_star,))	
	kk=0
	for ss in range(Dyn.MS.n_surf):
		Mstar=Dyn.MS.MM_star[ss]
		Nstar=Dyn.MS.NN_star[ss]
		for mm in range(Mstar):
			gammaw_ref[kk:kk+Nstar]=Sta.Gamma[ss][-1,:]
			kk+=Nstar

	er=np.max(np.abs( xsta[Dyn.K:Dyn.K+Dyn.K_star]-gammaw_ref ))
	print('Error wake circulation: %.3e' %er)
	assert er<1e-13, 'Steady-state gamma_star not matching!'

	er=np.max(np.abs( xsta[Dyn.K+Dyn.K_star:2*Dyn.K+Dyn.K_star] ))
	print('Error bound derivative: %.3e' %er)
	assert er<1e-13,'Non-zero derivative of circulation at steady state!'

	if Dyn.integr_order==2:
		er=np.max(np.abs( xsta[:Dyn.K]-xsta[-Dyn.K:] ))
		print('Error bound circulation previous vs current time-step: %.3e'%er )
		assert er<1e-13,\
					'Circulation at previous and current time-step not matching'








	












