'''
Geometrical methods for bound surface
S. Maraniello, 20 May 2018
'''

import ctypes as ct
import numpy as np
import itertools

# from IPython import embed
dmver=np.array([ 0, 1, 1, 0]) # delta to go from (m,n) panel to (m,n) vertices
dnver=np.array([ 0, 0, 1, 1])

from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils
import sharpy.linear.src.libuvlm as libuvlm

libc=ct_utils.import_ctypes_lib(SharpyDir + '/lib/', 'libuvlm')


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

		###
		# mpv=self.maps.from_panel_to_vertices(m,n)

		###
		# mpv=self.maps.Mpv[m,n,:,:]
		# zetav_here_ref=np.zeros((4,3),order='C')
		# for ii in range(4):
		# 	zetav_here_ref[ii,:]=self.zeta[:,mpv[ii,0],mpv[ii,1]]
		# zetav_here=self.zeta[:,dmver+m,dnver+n].T
		# assert np.max(np.abs(zetav_here-zetav_here_ref))<1e-16, embed()
		# return zetav_here

		###
		# return self.zeta[:,dmver+m,dnver+n].T

		return self.zeta[:, [m+0,m+1,m+1,m+0], [n+0,n+0,n+1,n+1]].T


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
		self.zetac=np.zeros((3,M,N),order='F') # F order avoids the need to copy
											   # when passing self.zetac[:,x,x] to
											   # C written libraries.

		for mm in range(M):
			for nn in range(N):
				zetav_here=self.get_panel_vertices_coords(mm,nn)
				self.zetac[:,mm,nn]=self.get_panel_collocation(zetav_here)


	# -------------------------------------------------- get mid-segment points

	def get_panel_wsv(self):
		pass


	def get_panel_midsegments(self,zetav_here):
		pass


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

	def __init__(self,Map,zeta,gamma,
							u_ext=None,zeta_dot=None,gamma_dot=None,
														  rho=1.,aM=0.5,aN=0.5,
														  omega=np.zeros((3),)):

		super().__init__(Map,zeta,aM,aN)

		self.gamma=gamma
		self.zeta_dot=zeta_dot
		self.u_ext=u_ext
		self.gamma_dot=gamma_dot
		self.rho=rho
		self.omega=omega

		msg_out='wrong input shape!'
		assert self.gamma.shape==(self.maps.M,self.maps.N), msg_out
		assert self.zeta.shape==(3,self.maps.M+1,self.maps.N+1), msg_out
		if self.zeta_dot is not None:
			assert self.zeta_dot.shape==(3,self.maps.M+1,self.maps.N+1), msg_out
		if self.u_ext is not None:
			assert self.u_ext.shape==(3,self.maps.M+1,self.maps.N+1), msg_out


	# -------------------------------------------------------- input velocities

	def get_input_velocities_at_collocation_points(self):
		'''
		Returns velocities at collocation points from nodal values u_ext and
		zeta_dot of shape (3,M+1,N+1).

		Remark: u_input_coll=Wcv*(u_ext-zet_dot) does not depend on the
		coordinates zeta.

        2018/08/24: Include effects due to rotation (omega x zeta). Now it
        depends on the coordinates zeta
		'''

		# define total velocity
		if self.zeta_dot is not None:
			u_tot=self.u_ext-self.zeta_dot
		else:
			u_tot=self.u_ext

		# Include rotation
		for i_m in range(self.maps.M+1):
			for i_n in range(self.maps.N+1):
				#print(u_tot[:,i_m,i_n])
				#print(np.cross(self.omega,self.zeta[:,i_m,i_n]))
				u_tot[:,i_m,i_n] -= np.cross(self.omega,self.zeta[:,i_m,i_n])

		self.u_input_coll=self.interp_vertex_to_coll(u_tot)


	def get_normal_input_velocities_at_collocation_points(self):
		'''
		From nodal input velocity to normal velocities at collocation points.
		'''

		#M,N=self.maps.M,self.maps.N

		# produce velocities at collocation points
		self.get_input_velocities_at_collocation_points()
		self.u_input_coll_norm=self.project_coll_to_normal(self.u_input_coll)



	def get_input_velocities_at_segments(self):
		'''
		Returns velocities at mid-segment points from nodal values u_ext and
		zeta_dot of shape (3,M+1,N+1).

		Warning: input velocities at grid segments are stored in a redundant
		format:
			(3,4,M,N)
		where the element
			(:,ss,mm,nn)
		is the induced velocity over the ss-th segment of panel (mm,nn). A fast
		looping is implemented to re-use previously computed velocities

        2018/08/24: Include effects due to rotation (omega x zeta). Now it
        depends on the coordinates zeta
		'''

		# define total velocity
		if self.zeta_dot is not None:
			u_tot=self.u_ext-self.zeta_dot
		else:
			u_tot=self.u_ext

        # Include rotation
		for i_m in range(self.maps.M+1):
			for i_n in range(self.maps.N+1):
				u_tot[:,i_m,i_n] -= np.cross(self.omega,self.zeta[:,i_m,i_n])

		M,N=self.maps.M,self.maps.N
		self.u_input_seg=np.empty((3,4,M,N))


		# indiced as per self.maps
		dmver=[ 0, 1, 1, 0] # delta to go from (m,n) panel to (m,n) vertices
		dnver=[ 0, 0, 1, 1]
		svec =[ 0, 1, 2, 3] # seg. no.
		avec =[ 0, 1, 2, 3] # 1st vertex no.
		bvec =[ 1, 2, 3, 0] # 2nd vertex no.

		##### panel (0,0): compute all
		mm,nn=0,0
		for ss,aa,bb in zip(svec,avec,bvec):
			uA=u_tot[:,mm+dmver[aa],nn+dnver[aa]]
			uB=u_tot[:,mm+dmver[bb],nn+dnver[bb]]
			self.u_input_seg[:,ss,mm,nn]=.5*(uA+uB)

		##### panels n=0: copy seg.3
		nn=0
		svec=[0,1,2] # seg. no.
		avec=[0,1,2] # 1st vertex no.
		bvec=[1,2,3] # 2nd vertex no.
		for mm in range(1,M):
			for ss,aa,bb in zip(svec,avec,bvec):
				uA=u_tot[:,mm+dmver[aa],nn+dnver[aa]]
				uB=u_tot[:,mm+dmver[bb],nn+dnver[bb]]
				self.u_input_seg[:,ss,mm,nn]=.5*(uA+uB)
			self.u_input_seg[:,3,mm,nn]=self.u_input_seg[:,1,mm-1,nn]
		##### panels m=0: copy seg.0
		mm=0
		svec=[1,2,3] # seg. number
		avec=[1,2,3] # 1st vertex of seg.
		bvec=[2,3,0] # 2nd vertex of seg.
		for nn in range(1,N):
			for ss,aa,bb in zip(svec,avec,bvec):
				uA=u_tot[:,mm+dmver[aa],nn+dnver[aa]]
				uB=u_tot[:,mm+dmver[bb],nn+dnver[bb]]
				self.u_input_seg[:,ss,mm,nn]=.5*(uA+uB)
			self.u_input_seg[:,0,mm,nn]=self.u_input_seg[:,2,mm,nn-1]
		##### all others: copy seg. 0 and 3
		svec=[1,2] # seg. number
		avec=[1,2] # 1st vertex of seg.
		bvec=[2,3] # 2nd vertex of seg.
		for pp in itertools.product(range(1,M),range(1,N)):
			mm,nn=pp
			for ss,aa,bb in zip(svec,avec,bvec):
				uA=u_tot[:,mm+dmver[aa],nn+dnver[aa]]
				uB=u_tot[:,mm+dmver[bb],nn+dnver[bb]]
				self.u_input_seg[:,ss,mm,nn]=.5*(uA+uB)
			self.u_input_seg[:,0,mm,nn]=self.u_input_seg[:,2,mm,nn-1]
			self.u_input_seg[:,3,mm,nn]=self.u_input_seg[:,1,mm-1,nn]

		return self



	# ------------------------------------------------------ induced velocities

	def get_induced_velocity(self,zeta_target):
		'''
		Computes induced velocity at a point zeta_target.
		'''

		M,N=self.maps.M,self.maps.N
		uind_target=np.zeros((3,),order='C')
		#uind_ref=np.zeros((3,),order='C')

		for mm in range(M):
			for nn in range(N):
				# panel info
				zetav_here=self.get_panel_vertices_coords(mm,nn)
				uind_target+=libuvlm.biot_panel_cpp(zeta_target,
												   zetav_here,self.gamma[mm,nn])

		return uind_target



	def get_induced_velocity_cpp(self,zeta_target):
		'''
		Computes induced velocity at a point zeta_target.
		'''

		assert zeta_target.flags['C_CONTIGUOUS'], "Input not C contiguous"

		M,N=self.maps.M,self.maps.N
		uind_target=np.zeros((3,),order='C')

		libc.call_ind_vel(
			uind_target.ctypes.data_as(ct.POINTER(ct.c_double)),
			zeta_target.ctypes.data_as(ct.POINTER(ct.c_double)),
			self.zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
			self.gamma.ctypes.data_as(ct.POINTER(ct.c_double)),
			ct.byref(ct.c_int(M)),
			ct.byref(ct.c_int(N)))

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
			aic3[:,cc]=libuvlm.biot_panel_cpp(zeta_target,zetav_here,gamma=1.0)

		return aic3


	def get_aic3_cpp(self,zeta_target):
		'''
		Produces influence coefficinet matrix to calculate the induced velocity
		at a target point. The aic3 matrix has shape (3,K)
		'''

		assert zeta_target.flags['C_CONTIGUOUS'], "Input not C contiguous"

		K=self.maps.K
		aic3=np.zeros((3,K),order='C')

		libc.call_aic3(
			aic3.ctypes.data_as(ct.POINTER(ct.c_double)),
			zeta_target.ctypes.data_as(ct.POINTER(ct.c_double)),
			self.zeta.ctypes.data_as(ct.POINTER(ct.c_double)),
			ct.byref(ct.c_int(self.maps.M)),
			ct.byref(ct.c_int(self.maps.N)))

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

		Warning: induced velocities at grid segments are stored in a redundant
		format:
			(3,4,M,N)
		where the element
			(:,ss,mm,nn)
		is the induced velocity over the ss-th segment of panel (mm,nn). A fast
		looping is implemented to re-use previously computed velocities
		'''

		M_trg=Surf_target.maps.M
		N_trg=Surf_target.maps.N

		if target=='collocation':
			if not hasattr(Surf_target,'zetac'):
				Surf_target.generate_collocations()
			ZetaTarget=Surf_target.zetac

			if Project:
				if not hasattr(Surf_target,'normals'):
					Surf_target.generate_normals()
				Uind=np.empty((M_trg,N_trg))
			else:
				Uind=np.empty((3,M_trg,N_trg))

			# loop target points
			for pp in itertools.product(range(M_trg),range(N_trg)):
				mm,nn=pp
				uind=self.get_induced_velocity_cpp(ZetaTarget[:,mm,nn])
				if Project:
					Uind[mm,nn]=np.dot(uind,Surf_target.normals[:,mm,nn])
				else:
					Uind[:,mm,nn]=uind

		if target=='segments':

			if Project:
				raise NameError('Normal not defined for segment')

			Uind=np.zeros((3,4,M_trg,N_trg))

			##### panel (0,0): compute all
			mm,nn=0,0
			svec=[0,1,2,3] # seg. number
			avec=[0,1,2,3] # 1st vertex of seg.
			bvec=[1,2,3,0] # 2nd vertex of seg.
			zetav_here=Surf_target.get_panel_vertices_coords(mm,nn)
			for ss,aa,bb in zip(svec,avec,bvec):
				zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
				Uind[:,ss,mm,nn]=self.get_induced_velocity_cpp(zeta_mid)

			##### panels n=0: copy seg.3
			nn=0
			svec=[0,1,2] # seg. number
			avec=[0,1,2] # 1st vertex of seg.
			bvec=[1,2,3] # 2nd vertex of seg.
			for mm in range(1,M_trg):
				zetav_here=Surf_target.get_panel_vertices_coords(mm,nn)
				for ss,aa,bb in zip(svec,avec,bvec):
					zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
					Uind[:,ss,mm,nn]=self.get_induced_velocity_cpp(zeta_mid)
				Uind[:,3,mm,nn]=Uind[:,1,mm-1,nn]

			##### panels m=0: copy seg.0
			mm=0
			svec=[1,2,3] # seg. number
			avec=[1,2,3] # 1st vertex of seg.
			bvec=[2,3,0] # 2nd vertex of seg.
			for nn in range(1,N_trg):
				zetav_here=Surf_target.get_panel_vertices_coords(mm,nn)
				for ss,aa,bb in zip(svec,avec,bvec):
					zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
					Uind[:,ss,mm,nn]=self.get_induced_velocity_cpp(zeta_mid)
				Uind[:,0,mm,nn]=Uind[:,2,mm,nn-1]

			##### all others: copy seg. 0 and 3
			svec=[1,2] # seg. number
			avec=[1,2] # 1st vertex of seg.
			bvec=[2,3] # 2nd vertex of seg.
			for pp in itertools.product(range(1,M_trg),range(1,N_trg)):
				mm,nn=pp
				zetav_here=Surf_target.get_panel_vertices_coords(*pp)
				for ss,aa,bb in zip(svec,avec,bvec):
					zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
					Uind[:,ss,mm,nn]=self.get_induced_velocity_cpp(zeta_mid)
				Uind[:,0,mm,nn]=Uind[:,2,mm,nn-1]
				Uind[:,3,mm,nn]=Uind[:,1,mm-1,nn]

		return Uind



	def get_aic_over_surface(self,Surf_target,
											 target='collocation',Project=True):
		'''
		Produces influence coefficient matrices such that the velocity induced
		over the Surface_target is given by the product:

		if target=='collocation':
			if Project:
				u_ind_coll_norm.rehape(-1)=AIC*self.gamma.reshape(-1,order='C')
			else:
				u_ind_coll_norm[ii,:,:].rehape(-1)=
									AIC[ii,:,:]*self.gamma.reshape(-1,order='C')
				where ii=0,1,2

		if targer=='segments':
			- AIC has shape (3,self.maps.K,4,Mout,Nout), such that
				AIC[:,:,ss,mm,nn]
			is the influence coefficient matrix associated to the induced
			velocity at segment ss of panel (mm,nn)
		'''

		K_in=self.maps.K

		if target=='collocation':

			K_out=Surf_target.maps.K
			if not hasattr(Surf_target,'zetac'):
				Surf_target.generate_collocations()
			ZetaTarget=Surf_target.zetac

			if Project:
				if not hasattr(Surf_target,'normals'):
					Surf_target.generate_normals()
				AIC=np.empty((K_out,K_in))
			else:
				AIC=np.empty((3,K_out,K_in))

			# loop target points
			for cc in range(K_out):
				# retrieve panel coords
				mm=Surf_target.maps.ind_2d_pan_scal[0][cc]
				nn=Surf_target.maps.ind_2d_pan_scal[1][cc]
				# retrieve influence coefficients
				#ref_aic3=self.get_aic3(ZetaTarget[:,mm,nn])
				aic3=self.get_aic3_cpp(ZetaTarget[:,mm,nn])
				#assert np.max(np.abs(aic3-ref_aic3))<1e-13, embed()

				if Project:
					AIC[cc,:]=np.dot(Surf_target.normals[:,mm,nn],aic3)
				else:
					AIC[:,cc,:]=aic3


		if target=='segments':
			if Project:
				raise NameError('Normal not defined at collocation points')

			M_trg,N_trg=Surf_target.maps.M,Surf_target.maps.N
			AIC=np.zeros((3,K_in,4,M_trg,N_trg))

			##### panel (0,0): compute all
			mm,nn=0,0
			svec=[0,1,2,3] # seg. number
			avec=[0,1,2,3] # 1st vertex of seg.
			bvec=[1,2,3,0] # 2nd vertex of seg.
			zetav_here=Surf_target.get_panel_vertices_coords(mm,nn)
			for ss,aa,bb in zip(svec,avec,bvec):
				zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
				AIC[:,:,ss,mm,nn]=self.get_aic3_cpp(zeta_mid)

			##### panels n=0: copy seg.3
			nn=0
			svec=[0,1,2] # seg. number
			avec=[0,1,2] # 1st vertex of seg.
			bvec=[1,2,3] # 2nd vertex of seg.
			for mm in range(1,M_trg):
				zetav_here=Surf_target.get_panel_vertices_coords(mm,nn)
				for ss,aa,bb in zip(svec,avec,bvec):
					zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
					AIC[:,:,ss,mm,nn]=self.get_aic3_cpp(zeta_mid)
				AIC[:,:,3,mm,nn]=AIC[:,:,1,mm-1,nn]

			##### panels m=0: copy seg.0
			mm=0
			svec=[1,2,3] # seg. number
			avec=[1,2,3] # 1st vertex of seg.
			bvec=[2,3,0] # 2nd vertex of seg.
			for nn in range(1,N_trg):
				zetav_here=Surf_target.get_panel_vertices_coords(mm,nn)
				for ss,aa,bb in zip(svec,avec,bvec):
					zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
					AIC[:,:,ss,mm,nn]=self.get_aic3_cpp(zeta_mid)
				AIC[:,:,0,mm,nn]=AIC[:,:,2,mm,nn-1]

			##### all others: copy seg. 0 and 3
			svec=[1,2] # seg. number
			avec=[1,2] # 1st vertex of seg.
			bvec=[2,3] # 2nd vertex of seg.
			for pp in itertools.product(range(1,M_trg),range(1,N_trg)):
				mm,nn=pp
				zetav_here=Surf_target.get_panel_vertices_coords(*pp)
				for ss,aa,bb in zip(svec,avec,bvec):
					zeta_mid=0.5*(zetav_here[aa,:]+zetav_here[bb,:])
					AIC[:,:,ss,mm,nn]=self.get_aic3_cpp(zeta_mid)
				AIC[:,:,3,mm,nn]=AIC[:,:,1,mm-1,nn]
				AIC[:,:,0,mm,nn]=AIC[:,:,2,mm,nn-1]

		return AIC


	# ------------------------------------------------------------------ forces

	def get_joukovski_qs(self,gammaw_TE=None,recompute_velocities=True):
		'''
		Returns quasi-steady forces evaluated at mid-segment points over the
		surface.

		Important: the circulation at the first row of wake panel is required!
		Hence all

		Warning: forces are stored in a NON-redundant format:
			(3,4,M,N)
		where the element
			(:,ss,mm,nn)
		is the contribution to the force over the ss-th segment due to the
		circulation of panel (mm,nn).
		'''

		if not hasattr(self,'u_input_seg'):
			if not recompute_velocities:
				print("WARNING: recomputing velocities")
			self.get_input_velocities_at_segments()
		if not hasattr(self,'u_ind_seg'):
			raise NameError('u_ind_seg not available!')

		M,N=self.maps.M,self.maps.N
		self.fqs_seg_unit=np.zeros((3,4,M,N))
		self.fqs=np.zeros((3,M+1,N+1))

		# indiced as per self.maps
		dmver=[ 0, 1, 1, 0] # delta to go from (m,n) panel to (m,n) vertices
		dnver=[ 0, 0, 1, 1]
		svec =[ 0, 1, 2, 3] # seg. no.
		avec =[ 0, 1, 2, 3] # 1st vertex no.
		bvec =[ 1, 2, 3, 0] # 2nd vertex no.

		### force produced by BOUND panels
		for pp in itertools.product(range(0,M),range(0,N)):
			mm,nn=pp
			zetav_here=self.get_panel_vertices_coords(mm,nn)
			for ss,aa,bb in zip(svec,avec,bvec):
				df=libuvlm.joukovski_qs_segment(
					zetaA=zetav_here[aa,:],zetaB=zetav_here[bb,:],
					v_mid=self.u_ind_seg[:,ss,mm,nn]+self.u_input_seg[:,ss,mm,nn],
					gamma=1.0,fact=self.rho)
				self.fqs_seg_unit[:,ss,mm,nn]=df
				# project on vertices
				self.fqs[:,mm+dmver[aa],nn+dnver[aa]]+=0.5*self.gamma[mm,nn]*df
				self.fqs[:,mm+dmver[bb],nn+dnver[bb]]+=0.5*self.gamma[mm,nn]*df

		### force produced by wake T.E. segments
		# Note:
		# 1. zetaA & zetaB are ordered such that the wake circulation is
		# subtracts to the bound circulation over TE segment
		# 2. the TE segment corresponds to seg.1 of the last row of BOUND panels
		if gammaw_TE is None:
			raise NameError('Enter gammaw_TE - option disabled for debugging')
			gammaw_TE=self.gamma[M-1,:]

		self.fqs_wTE_unit=np.zeros((3,N))

		for nn in range(N):
			df=libuvlm.joukovski_qs_segment(
					zetaA=self.zeta[:,M,nn+1],
					zetaB=self.zeta[:,M,nn],
					v_mid=self.u_input_seg[:,1,M-1,nn]+self.u_ind_seg[:,1,M-1,nn],
					gamma=1.0,
					fact=self.rho)
			# record force on TE due to wake and project
			self.fqs_wTE_unit[:,nn]=df
			self.fqs[:,M,nn+1]+=0.5*gammaw_TE[nn]*df
			self.fqs[:,M,nn]+=0.5*gammaw_TE[nn]*df

		return self



	def get_joukovski_unsteady(self):
		'''
		Returns added mass effects over lattive grid
		'''

		if self.gamma_dot is None:
			raise NameError('circulation derivative not specified!')
		if not hasattr(self,'areas'):
			self.generate_areas()

		M,N=self.maps.M,self.maps.N
		self.funst=np.zeros((3,M+1,N+1))
		wcv=self.get_panel_wcv()

		for pp in itertools.product(range(0,M),range(0,N)):
			mm,nn=pp

			# force at collocation point
			fcoll=-self.rho*self.areas[mm,nn]*self.normals[:,mm,nn]*self.gamma_dot[mm,nn]


			# project at vertices
			for vv in range(4):
				self.funst[:,mm+dmver[vv],nn+dnver[vv]]+=wcv[vv]*fcoll




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

	S=AeroGridSurface(Map,zeta=tsdata.zeta[ss],
								gamma=tsdata.gamma[ss],
										zeta_dot=tsdata.zeta_dot[ss],
												u_ext=tsdata.u_ext[ss],
													 gamma_dot=tsdata.gamma_dot[ss])
	S.get_normal_input_velocities_at_collocation_points()

	# verify aic3
	zeta_out=np.array([1,4,2])
	uind_out=S.get_induced_velocity_cpp(zeta_out)
	aic3=S.get_aic3_cpp(zeta_out)
	uind_out2=np.dot(aic3,S.gamma.reshape(-1,order='C'))
	assert np.max(np.abs(uind_out-uind_out2))<1e-12, 'Wrong aic3 calculation'

	# calc unsteady joukovski force
	S.generate_areas()
	S.get_joukovski_unsteady()
