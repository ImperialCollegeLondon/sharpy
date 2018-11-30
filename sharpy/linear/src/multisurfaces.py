'''
Linearise UVLM
S. Maraniello, 25 May 2018
'''

import numpy as np
import sharpy.linear.src.libuvlm as libuvlm
import sharpy.linear.src.gridmapping as gridmapping
import sharpy.linear.src.surface as surface
import sharpy.linear.src.assembly as assembly


class MultiAeroGridSurfaces():
	'''
	Creates and assembles multiple aerodynamic surfaces from data
	'''

	def __init__(self,tsdata,omega=np.zeros((3),)):
		'''
		Initialise rom data structure at time step.
		omega: rotation speed of the A FoR [rad/s]
		'''

		self.tsdata0=tsdata
		self.n_surf=tsdata.n_surf
		self.dimensions=tsdata.dimensions
		self.dimensions_star=tsdata.dimensions_star

		# allocate surfaces
		self.Surfs=[]
		self.Surfs_star=[]

		# allocate size lists - useful for global assembly
		self.NN=[]
		self.MM=[]
		self.KK=[]
		self.KKzeta=[]
		self.NN_star=[]
		self.MM_star=[]
		self.KK_star=[]
		self.KKzeta_star=[]

		for ss in range(self.n_surf):

			### Allocate bound surfaces
			M,N=tsdata.dimensions[ss]
			Map=gridmapping.AeroGridMap(M,N)
			Surf=surface.AeroGridSurface(
					Map,zeta=tsdata.zeta[ss],gamma=tsdata.gamma[ss],
					u_ext=tsdata.u_ext[ss],zeta_dot=tsdata.zeta_dot[ss],
					gamma_dot=tsdata.gamma_dot[ss],
					rho=tsdata.rho,
					omega=omega)
			# generate geometry data
			Surf.generate_areas()
			Surf.generate_normals()
			Surf.aM,Surf.aN=0.5,0.5
			Surf.generate_collocations()
			self.Surfs.append(Surf)
			# store size
			self.MM.append(M)
			self.NN.append(N)
			self.KK.append(Map.K)
			self.KKzeta.append(Map.Kzeta)

			### Allocate wake surfaces
			M,N=tsdata.dimensions_star[ss]
			Map=gridmapping.AeroGridMap(M,N)
			Surf=surface.AeroGridSurface(Map,
						  zeta=tsdata.zeta_star[ss],gamma=tsdata.gamma_star[ss],
						  rho=tsdata.rho)
			self.Surfs_star.append(Surf)
			# store size
			self.MM_star.append(M)
			self.NN_star.append(N)
			self.KK_star.append(Map.K)
			self.KKzeta_star.append(Map.Kzeta)


	def get_ind_velocities_at_collocation_points(self):
		'''
		Computes normal induced velocities at collocation points.
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


	def  get_input_velocities_at_collocation_points(self):

		for ss in range(self.n_surf):
			Surf=self.Surfs[ss]
			if not hasattr(Surf,'u_input_coll'):
				Surf.get_input_velocities_at_collocation_points()


	# -------------------------------------------------------------------------


	def get_ind_velocities_at_segments(self,overwrite=False):
		'''
		Computes induced velocities at mid-segment points.
		'''

		# Loop surfaces (where ind. velocity is computed)
		for ss_out in range(self.n_surf):

			Surf_out=self.Surfs[ss_out]
			if hasattr(Surf_out,'u_ind_seg') and (not overwrite):
				continue

			M_out,N_out=self.dimensions[ss_out]
			Surf_out.u_ind_seg=np.zeros((3,4,M_out,N_out))

			# Loop input surfaces
			for ss_in in range(self.n_surf):

				# Buond
				Surf_in=self.Surfs[ss_in]
				Surf_out.u_ind_seg+=\
					Surf_in.get_induced_velocity_over_surface(Surf_out,
											    target='segments',Project=False)

				# Wake
				Surf_in=self.Surfs_star[ss_in]
				Surf_out.u_ind_seg+=\
					Surf_in.get_induced_velocity_over_surface(Surf_out,
											    target='segments',Project=False)


	def  get_input_velocities_at_segments(self,overwrite=False):

		for ss in range(self.n_surf):
			Surf=self.Surfs[ss]
			if hasattr(Surf,'u_input_seg') and (not overwrite):
				continue
			Surf.get_input_velocities_at_segments()

	# -------------------------------------------------------------------------


	def get_joukovski_qs(self,overwrite=False):
		'''
		Returns quasi-steady forces over

		Warning: forces are stored in a NON-redundant format:
			(3,4,M,N)
		where the element
			(:,ss,mm,nn)
		is the contribution to the force over the ss-th segment due to the
		circulation of panel (mm,nn).

		'''

		# get input and induced velocities at segments
		self.get_input_velocities_at_segments(overwrite)
		self.get_ind_velocities_at_segments(overwrite)

		for ss in range(self.n_surf):
			Surf=self.Surfs[ss]
			Surf.get_joukovski_qs(gammaw_TE=self.Surfs_star[ss].gamma[0,:])


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

		print('Verifing non-penetration at bound...')
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


		print('Verifing AICs at collocation points...')
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



	def verify_joukovski_qs(self):
		'''
		Verify quasi-steady contribution for forces matches against SHARPy.
		'''

		print('Verifing joukovski quasi-steady forces...')
		self.get_joukovski_qs()

		for ss in range(self.n_surf):
			Surf=self.Surfs[ss]

			Fhere=Surf.fqs.reshape((3,Surf.maps.Kzeta))
			Fref=self.tsdata0.forces[ss][0:3].reshape((3,Surf.maps.Kzeta))
			# Check if forces and ct_forces_list are the same:
			# Fref_check=np.array(self.tsdata0.ct_forces_list[6*ss:6*ss+3])
			# print('Check forces: ', Fref_check-Fref)
			ErMax=np.max(np.abs(Fhere-Fref))

			print('Surface %.2d max abs error: %.3e' %(ss,ErMax) )
			assert ErMax<1e-12 ,'Wrong quasi-steady force over surface %.2d!'%ss





if __name__=='__main__':

	import read
	import matplotlib.pyplot as plt

	# select test case
	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
	#fname='../test/h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
	haero=read.h5file(fname)
	tsdata=haero.ts00000

	MS=MultiAeroGridSurfaces(tsdata)

	# collocation points
	MS.get_normal_ind_velocities_at_collocation_points()
	MS.verify_non_penetration()
	MS.verify_aic_coll()

	# joukovski
	MS.verify_joukovski_qs()



	embed()

	### verify u_induced
