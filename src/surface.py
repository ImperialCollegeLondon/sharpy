'''
Geometrical methods for bound surface
S. Maraniello, 20 May 2018
'''

import numpy as np
import libuvlm
from IPython import embed

#import gridmapping
#class Surface(gridmapping.AeroGridMap):
	# '''
	# Class for bound surface.
	# Requires a mapping structure
	# '''
	# def __init__(self,zeta,n_surf,dimensions):
	# 	super().__init__(n_surf,dimensions)



class AeroGridGeo():
	'''
	Allows retrieving geometrical information of a surface. Requires a 
	gridmapping.AeroGridMap mapping structure in input and the surface vertices 
	coordinates.

	Indices convention: each panel is characterised through the following
	indices:
	- m,n: chord/span-wise indices

	Methods:
	- get_*: retrieve information of a panel (e.g. normal, surface area)
	- generate_*: apply get_* method to each panel and store info into array.

	Interpolation matrices, W:
	- these are labelled as 'Wba', where 'a' defines the initial format, b the 
	final. Hence, given the array vb, it holds va=Wab*vb 
	
	'''
	def __init__(self,
				 Map:'gridmapping.AeroGridMap instance',
				 zeta:'Array of vertex coordinates at each surface',
				 aM:'chord-wise position of collocation point in panel'=0.5,
				 aN:'span-wise position of collocation point in panel'=0.5):

		self.maps=Map
		self.maps.map_all()
		self.zeta=zeta 
		self.aM=aM
		self.aN=aN

		### Mapping coefficients
		#self.wvc=self.get_panel_wcv(self.aM,self.aN)


	# -------------------------------------------------------------------------	

	def get_panel_vertices_coords(self,m,n):
		'''
		Retrieves coordinates of panel (m,n) vertices.
		'''

		#mpv=self.maps.from_panel_to_vertices(m,n)
		mpv=self.maps.Mpv[m,n,:,:]
		zetav_here=np.zeros((4,3))
		for ii in range(4):
			zetav_here[ii,:]=self.zeta[:,mpv[ii,0],mpv[ii,1]]

		return zetav_here


	# ------------------------------------------------------- get panel normals	

	def generate_normals(self):

		M,N=self.maps.M,self.maps.N
		self.normals=np.zeros((3,M,N))

		for mm in range(M):
			for nn in range(N):
				zetav_here=self.get_panel_vertices_coords(mm,nn)
				self.normals[:,mm,nn]=libuvlm.panel_normal(zetav_here)


	# -------------------------------------------------- get panel surface area

	def generate_areas(self):

		M,N=self.maps.M,self.maps.N
		self.areas=np.zeros((M,N))

		for mm in range(M):
			for nn in range(N):
				zetav_here=self.get_panel_vertices_coords(mm,nn)
				self.areas[mm,nn]=libuvlm.panel_area(zetav_here)


	# -------------------------------------------------- get collocation points

	def get_panel_wcv(self):
		'''
		Produces a compact array with weights for bilinear interpolation, where
		aN,aM in [0,1] are distances in the chordwise and spanwise directions 
		such that:
			- (aM,aN)=(0,0) --> quantity at vertex 0
			- (aM,aN)=(1,0) --> quantity at vertex 1
			- (aM,aN)=(1,1) --> quantity at vertex 2
			- (aM,aN)=(0,1) --> quantity at vertex 3
		'''

		aM=self.aM
		aN=self.aN
		wcv=np.array([ (1-aM)*(1-aN), aM*(1-aN), aM*aN, aN*(1-aM) ])

		return wcv 

	def get_panel_collocation(self,zetav_here):
		'''
		Using bilinear interpolation, retrieves panel collocation point, where
		aN,aM in [0,1] are distances in the chordwise and spanwise directions 
		such that:
			- (aM,aN)=(0,0) --> quantity at vertex 0
			- (aM,aN)=(1,0) --> quantity at vertex 1
			- (aM,aN)=(1,1) --> quantity at vertex 2
			- (aM,aN)=(0,1) --> quantity at vertex 3
		'''

		wcv=self.get_panel_wcv()
		zetac_here=np.dot(wcv,zetav_here)

		return zetac_here

	def generate_collocations(self):

		M,N=self.maps.M,self.maps.N
		self.zetac=np.zeros((3,M,N))

		for mm in range(M):
			for nn in range(N):
				zetav_here=self.get_panel_vertices_coords(mm,nn)
				self.zetac[:,mm,nn]=self.get_panel_collocation(zetav_here)
					

	# -------------------------------------------------- get mid-segment points

	def get_panel_wsv(self):
		pass

		#return wsv 


	def get_panel_midsegments(self,zetav_here):
		'''
		'''
		pass

		# wsv=self.get_panel_wsv()
		# zetac_here=np.dot(wcv,zetav_here)

		# return zetac_here

	def generate_midsegments():
		pass


	def generate_Wsv():
		pass



	# ----------------------------------------------- Interpolations/Projection

	def interp_vertex_to_coll(self,q_vert):
		'''
		Project a quantity q_vert (scalar or vector) defined at vertices to 
		collocation points.
		'''

		M,N=self.maps.M,self.maps.N
		#embed()
		inshape=q_vert.shape
		assert inshape[-2]==M+1 and inshape[-1]==N+1, 'Unexpected shape of q_vert'

		# determine weights
		wcv=self.get_panel_wcv()

		if len(inshape)==2:
			q_coll=np.zeros((M,N))
			for mm in range(M):
				for nn in range(N):
					# get q_vert at panel corners
					mpv=self.maps.from_panel_to_vertices(mm,nn)
					for vv in range(4):
						q_coll[mm,nn]=q_coll[mm,nn]+\
										     wcv[vv]*q_vert[mpv[vv,0],mpv[vv,1]]

		elif len(inshape)==3:
			q_coll=np.zeros((3,M,N))
			for mm in range(M):
				for nn in range(N):
					# get q_vert at panel corners
					mpv=self.maps.from_panel_to_vertices(mm,nn)
					for vv in range(4):
						q_coll[:,mm,nn]=q_coll[:,mm,nn]+\
										   wcv[vv]*q_vert[:,mpv[vv,0],mpv[vv,1]]
		else:
			raise NameError('Unexpected shape of q_vert')

		return q_coll


	def project_coll_to_normal(self,q_coll):
		'''
		Project a vector quantity q_coll defined at collocation points to normal.
		'''

		M,N=self.maps.M,self.maps.N
		assert q_coll.shape==(3,M,N) , 'Unexpected shape of q_coll'

		if not hasattr(self,'normals'):
			self.generate_normals()

		q_proj=np.zeros((M,N))
		for mm in range(M):
			for nn in range(N):
				q_proj[mm,nn]=np.dot(self.normals[:,mm,nn], q_coll[:,mm,nn])

		return q_proj


	# ------------------------------------------------------- visualise surface

	def plot(self,plot_normals=False):


		from mpl_toolkits.mplot3d import axes3d
		import matplotlib.pyplot as plt

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# Plot vertices grid
		ax.plot_wireframe(self.zeta[0],self.zeta[1],self.zeta[2]) 
		#rstride=10, cstride=10)
		
		# Plot collocation points
		ax.scatter(self.zetac[0],self.zetac[1],self.zetac[2],zdir='z',s=3,c='r')

		if plot_normals:
			ax.quiver(self.zetac[0],self.zetac[1],self.zetac[2],
						self.normals[0],self.normals[1],self.normals[2],
								 length=0.01*np.max(self.zeta),)#normalize=True)

		self.ax=ax





class AeroGridSurface(AeroGridGeo):
	'''
	Contains geometrical and aerodynamical information about bound/wake surface.

	Compulsory input are those that apply to both bound and wake surfaces:
	- zeta: defines geometry
	- gamma: circulation

	With respect to AeroGridGeo, the class contains methods to:
	- project prescribed input velocity at nodes (u_ext, zeta_dot) over 
	collocaiton points.
	- compute induced velocity over ANOTHER surface.
	- compute AIC induced over ANOTHER surface

	To add:
	- project prescribed input velocity at nodes (u_ext, zeta_dot) over 
	mid-point segments
	'''

	def __init__(self,Map,zeta,gamma,u_ext=None,zeta_dot=None,aM=0.5,aN=0.5):

		super().__init__(Map,zeta,aM,aN)

		self.gamma=gamma 
		self.zeta_dot=zeta_dot
		self.u_ext=u_ext

		msg_out='wrong input shape!'
		assert self.gamma.shape==(self.maps.M,self.maps.N), msg_out
		assert self.zeta.shape==(3,self.maps.M+1,self.maps.N+1), msg_out
		if self.zeta_dot is not None:
			assert self.zeta_dot.shape==(3,self.maps.M+1,self.maps.N+1), msg_out
		if self.u_ext is not None:
			assert self.u_ext.shape==(3,self.maps.M+1,self.maps.N+1), msg_out		



	# ---------------------------- project input velocity at collocation points

	def get_input_velocities_at_collocation_points(self):
		'''
		Returns velocities at collocation points from nodal values u_ext and
		zeta_dot of shape (3,M+1,N+1).

		Remark: u_input_coll=Wcv*(u_ext-zet_dot) does not depend on the 
		coordinates zeta.
		'''

		# define total velocity
		if self.zeta_dot is not None:
			u_tot=self.u_ext-self.zeta_dot
		else:
			u_tot=self.u_ext

		self.u_input_coll=self.interp_vertex_to_coll(u_tot)


	def get_normal_input_velocities_at_collocation_points(self):
		'''
		From nodal input velocity to normal velocities at collocation points.
		'''

		#M,N=self.maps.M,self.maps.N

		# produce velocities at collocation points
		self.get_input_velocities_at_collocation_points()
		self.u_input_coll_norm=self.project_coll_to_normal(self.u_input_coll)

		# # compute normal velocity at panels
		# if not hasattr(self,'normals'):
		# 	self.generate_normals()
		# self.u_input_coll_norm=np.zeros((M,N))
		# for mm in range(M):
		# 	for nn in range(N):
		# 		self.u_input_coll_norm[mm,nn]=np.dot(
		# 					   self.u_input_coll[:,mm,nn],self.normals[:,mm,nn])



	# ------------------------------- get normal velocity at collocation points

	def get_induced_velocity(self,zeta_target):
		'''
		Computes induced velocity at a point zeta_target.
		'''

		M,N=self.maps.M,self.maps.N
		uind_target=np.zeros(zeta_target.shape)

		for mm in range(M):
			for nn in range(N):
				# panel info
				zetav_here=self.get_panel_vertices_coords(mm,nn)
				uind_target+=libuvlm.biot_panel(zeta_target,
												   zetav_here,self.gamma[mm,nn])

		### Flag to remove last row???
		# ### remove last row contribution: segments 1->2
		# M_in,N_in=self.dimensions_star[ss_in]
		# for nn in range(N_in):			
		# 	mm=Min-1
		# 	gamma_here=Surf_in.gamma[mm,nn]							
		# 	zetav_here=Surf_in.get_panel_vertices_coords(mm,nn)
		# 	uindc-=libuvlm.biot_segment(zetac,zetav_here[3,:],zetav_here[0])

		return uind_target


	def get_aic3(self,zeta_target):
		'''
		Produces influence coefficinet matrix to calculate the induced velocity 
		at a target point. The aic3 matrix has shape (3,K)
		'''

		K=self.maps.K
		aic3=np.zeros((3,K))

		for cc in range(K):

			# define panel
			mm=self.maps.ind_2d_pan_scal[0][cc]
			nn=self.maps.ind_2d_pan_scal[1][cc]

			# get panel coordinates
			zetav_here=self.get_panel_vertices_coords(mm,nn)
			aic3[:,cc]=libuvlm.biot_panel(zeta_target,zetav_here,gamma=1.0)	

		return aic3


	def get_induced_velocity_over_surface(self,Surf_target,
											target='collocation',Project=False):
		'''
		Computes induced velocity over an instance of AeroGridSurface, where
		target specifies the target grid (collocation or segments). If Project
		is True, velocities are projected onver panel normal (only available at
		collocation points). 

		Note: for state-equation, both projected and non-projected velocities at 
		the collocation points are required. Hence, it is suggested to use this
		method with Projection=False, and project afterwards.
		'''

		if target=='collocation':
			M_trg=Surf_target.maps.M
			N_trg=Surf_target.maps.N
			if not hasattr(Surf_target,'zetac'):
				Surf_target.generate_collocations()
			ZetaTarget=Surf_target.zetac

			if Project:
				if not hasattr(Surf_target,'normals'):
					Surf_target.generate_normals()
				Uind=np.empty((M_trg,N_trg))
			else:
				Uind=np.empty((3,M_trg,N_trg))			

		if target=='segments':
			if Project:
				raise NameError('Normal not defined at collocation points')
			raise NameError('Method not implemented for segments')

		# loop target points
		for mm in range(M_trg):
			for nn in range(N_trg):
				uind=self.get_induced_velocity(ZetaTarget[:,mm,nn])
				if Project:
					Uind[mm,nn]=np.dot(uind,Surf_target.normals[:,mm,nn])
				else:
					Uind[:,mm,nn]=uind

		return Uind



	def get_aic_over_surface(self,Surf_target,
											 target='collocation',Project=True):
		'''
		Produces influence coefficient matrices such that the velocity induced
		over the Surface_target is given by the product:

		if Project:
			u_ind_coll_norm.rehape(-1)=AIC*self.gamma.reshape(-1,order='C')
		else:
			u_ind_coll_norm[ii,:,:].rehape(-1)=
									AIC[ii,:,:]*self.gamma.reshape(-1,order='C')
			where ii=0,1,2
		'''
		
		K_in=self.maps.K

		if target=='collocation':
			K_out=Surf_target.maps.K
			#ind_1d_to_multi=
			if not hasattr(Surf_target,'zetac'):
				Surf_target.generate_collocations()
			ZetaTarget=Surf_target.zetac

			if Project:
				if not hasattr(Surf_target,'normals'):
					Surf_target.generate_normals()
				AIC=np.empty((K_out,K_in))
			else:
				AIC=np.empty((3,K_out,K_in))	

		if target=='segments':
			if Project:
				raise NameError('Normal not defined at collocation points')
			raise NameError('Method not implemented for segments')

		# loop target points
		for cc in range(K_out): 
			# retrieve panel coords
			mm=Surf_target.maps.ind_2d_pan_scal[0][cc]
			nn=Surf_target.maps.ind_2d_pan_scal[1][cc]
			# retrieve influence coefficients
			aic3=self.get_aic3(ZetaTarget[:,mm,nn])
			if Project:
				AIC[cc,:]=np.dot(Surf_target.normals[:,mm,nn],aic3)
			else:
				AIC[:,cc,:]=aic3

		return AIC



if __name__=='__main__':

	import read, gridmapping
	import matplotlib.pyplot as plt 

	# select test case
	fname='../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
	haero=read.h5file(fname)
	tsdata=haero.ts00000

	# select surface and retrieve data
	ss=0
	M,N=tsdata.dimensions[ss]
	Map=gridmapping.AeroGridMap(M,N)
	G=AeroGridGeo(Map,tsdata.zeta[ss])
	# generate geometry data
	G.generate_areas()
	G.generate_normals()
	G.generate_collocations()

	# Visualise
	G.plot(plot_normals=True)
	plt.close('all')
	#plt.show()

	S=AeroGridSurface(Map,zeta=tsdata.zeta[ss],gamma=tsdata.gamma[ss],
							zeta_dot=tsdata.zeta_dot[ss],u_ext=tsdata.u_ext[ss])
	S.get_normal_input_velocities_at_collocation_points()

	# verify aic3
	zeta_out=np.array([1,4,2])
	uind_out=S.get_induced_velocity(zeta_out)
	aic3=S.get_aic3(zeta_out)
	uind_out2=np.dot(aic3,S.gamma.reshape(-1,order='C'))
	assert np.max(np.abs(uind_out-uind_out2))<1e-12, 'Wrong aic3 calculation'







