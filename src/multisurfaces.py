'''
Linearise UVLM
S. Maraniello, 25 May 2018
'''

import numpy as np 
import libuvlm
import gridmapping, surface
import assembly
from IPython import embed


class MultiAeroGridSurfaces():
	'''
	Creates and assembles multiple aerodynamic surfaces from data
	'''

	def __init__(self,tsdata):
		'''
		Initialise rom data structure at time step.
		'''

		self.tsdata0=tsdata
		self.n_surf=tsdata.n_surf
		self.dimensions=tsdata.dimensions
		self.dimensions_star=tsdata.dimensions_star

		# allocate surfaces
		self.Surfs=[]
		self.Surfs_star=[]

		for ss in range(self.n_surf):

			### Allocate bound surfaces
			M,N=tsdata.dimensions[ss]
			Map=gridmapping.AeroGridMap(M,N)
			Surf=surface.AeroGridSurface(
					Map,zeta=tsdata.zeta[ss],gamma=tsdata.gamma[ss],
							u_ext=tsdata.u_ext[ss],zeta_dot=tsdata.zeta_dot[ss])
			# generate geometry data
			Surf.generate_areas()
			Surf.generate_normals()
			Surf.aM,Surf.aN=0.5,0.5
			Surf.generate_collocations()
			self.Surfs.append(Surf)

			### Allocate wake surfaces
			M,N=tsdata.dimensions_star[ss]
			Map=gridmapping.AeroGridMap(M,N)
			Surf=surface.AeroGridSurface(Map,
						  zeta=tsdata.zeta_star[ss],gamma=tsdata.gamma_star[ss])
			self.Surfs_star.append(Surf)


	def get_ind_velocities_at_collocation_points(self):
		'''
		Computes normal induced velocities at collocation points from nodal 
		values u_ext.
		'''

		# Loop surfaces (where ind. velocity is computed)
		for ss_out in range(self.n_surf):

			# define array
			Surf_out=self.Surfs[ss_out]
			M_out,N_out=self.dimensions[ss_out]
			Surf_out.u_ind_coll=np.zeros((3,M_out,N_out))

			# Loop input surfaces
			for ss_in in range(self.n_surf):

				# Buond
				Surf_in=self.Surfs[ss_in]
				Surf_out.u_ind_coll+=\
					Surf_in.get_induced_velocity_over_surface(Surf_out,
											 target='collocation',Project=False)

				# Wake
				Surf_in=self.Surfs_star[ss_in]
				Surf_out.u_ind_coll+=\
					Surf_in.get_induced_velocity_over_surface(Surf_out,
											 target='collocation',Project=False)



	def get_normal_ind_velocities_at_collocation_points(self):
		'''
		Computes normal induced velocities at collocation points. 

		Note: for state-equation both projected and not projected induced
		velocities are required at the collocation points. Hence, this method
		tries to first the u_ind_coll attribute in each surface.
		'''

		# Loop surfaces (where ind. velocity is computed)
		for ss_out in range(self.n_surf):

			# define array
			Surf_out=self.Surfs[ss_out]
			M_out,N_out=self.dimensions[ss_out]
			#Surf_out.u_ind_coll_norm=np.empty((M_out,N_out))
			Surf_out.u_ind_coll_norm=np.zeros((M_out,N_out))


			if hasattr(Surf_out,'u_ind_coll'):
				Surf_out.u_ind_coll_norm=\
							Surf_out.project_coll_to_normal(Surf_out.u_ind_coll)
				# embed()
				# Surf_out.u_ind_coll_norm=\
				# 			 Surf_out.interp_vertex_to_coll(Surf_out.u_ind_coll)
				# for mm in range(M_out):
				# 	for nn in range(N_out):
				# 		Surf_out.u_ind_coll_norm[mm,nn]=\
				# 				 np.dot( Surf_out.normals[:,mm,nn],
				# 								  Surf_out.u_ind_coll[:,mm,nn] )
			else:
				# Loop input surfaces
				for ss_in in range(self.n_surf):

					# Buond
					Surf_in=self.Surfs[ss_in]
					Surf_out.u_ind_coll_norm+=\
						Surf_in.get_induced_velocity_over_surface(Surf_out,
											  target='collocation',Project=True)

					# Wake
					Surf_in=self.Surfs_star[ss_in]
					Surf_out.u_ind_coll_norm+=\
						Surf_in.get_induced_velocity_over_surface(Surf_out,
											  target='collocation',Project=True)
					

	def verify_non_penetration(self):
		'''
		Verify state variables fulfill non-penetration condition at bound 
		surfaces 
		'''

		# verify induced velocities have been computed
		for ss in range(self.n_surf):
			if not hasattr(self.Surfs[ss],'u_ind_coll_norm'):
				self.get_normal_ind_velocities_at_collocation_points()
				break

		print('Verify non-penetration at bound...')
		for ss in range(self.n_surf):
			Surf_here=self.Surfs[ss]
			# project input velocities
			if not hasattr(Surf_here,'u_input_coll_norm'):
				Surf_here.get_normal_input_velocities_at_collocation_points()

			ErMax=np.max(np.abs(
						 Surf_here.u_ind_coll_norm+Surf_here.u_input_coll_norm))
			print('Surface %.2d max abs error: %.3e' %(ss,ErMax) )

			assert ErMax<1e-12*np.max(np.abs(self.Surfs[0].u_ext)),\
			'Linearisation state does not verify the non-penetration condition!'



	def verify_aic_coll(self):
		'''
		Verify aic at collocaiton points using non-penetration condition
		'''

		AIC_list, AIC_star_list=assembly.AICs(
				   self.Surfs,self.Surfs_star,target='collocation',Project=True)

		### Compute iduced velocity
		for ss_out in range(self.n_surf):
			Surf_out=self.Surfs[ss_out]
			Surf_out.u_ind_coll_norm=np.zeros((Surf_out.maps.K,))
			for ss_in in range(self.n_surf):
				# Bound surface
				Surf_in=self.Surfs[ss_in]
				aic=AIC_list[ss_out][ss_in]
				Surf_out.u_ind_coll_norm+=np.dot(
										aic,Surf_in.gamma.reshape(-1,order='C'))

				# Wakes
				Surf_in=self.Surfs_star[ss_in]
				aic=AIC_star_list[ss_out][ss_in]				
				Surf_out.u_ind_coll_norm+=np.dot(
										aic,Surf_in.gamma.reshape(-1,order='C'))

			Surf_out.u_ind_coll_norm=\
			 Surf_out.u_ind_coll_norm.reshape((Surf_out.maps.M,Surf_out.maps.N))


		print('Verify AICs at collocation points...')
		for ss in range(self.n_surf):
			Surf_here=self.Surfs[ss]
			# project input velocities
			if not hasattr(Surf_here,'u_input_coll_norm'):
				Surf_here.get_normal_input_velocities_at_collocation_points()

			ErMax=np.max(np.abs(
						 Surf_here.u_ind_coll_norm+Surf_here.u_input_coll_norm))
			print('Surface %.2d max abs error: %.3e' %(ss,ErMax) )

			assert ErMax<1e-12*np.max(np.abs(self.Surfs[0].u_ext)),\
			'Linearisation state does not verify the non-penetration condition!'




if __name__=='__main__':

	import read
	import gridmapping, surface
	import matplotlib.pyplot as plt 

	# select test case
	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
	fname='../test/h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
	haero=read.h5file(fname)
	tsdata=haero.ts00000

	MS=MultiAeroGridSurfaces(tsdata)
	MS.get_normal_ind_velocities_at_collocation_points()
	MS.verify_non_penetration()
	MS.verify_aic_coll()








