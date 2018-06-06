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
import read, gridmapping, surface, assembly
from IPython import embed


class Test_assembly(unittest.TestCase):
	'''
	Test methods into assembly module
	'''

	def setUp(self):

		# select test case
		fname='./h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
		haero=read.h5file(fname)
		self.tsdata=haero.ts00000

		# select surface and retrieve data
		self.ss=0
		M,N=self.tsdata.dimensions[self.ss]
		self.Map=gridmapping.AeroGridMap(M,N)
		self.Surf=surface.AeroGridSurface(self.Map,zeta=self.tsdata.zeta[self.ss],
												gamma=self.tsdata.gamma[self.ss])

		# generate geometry data
		self.Surf.generate_areas()
		self.Surf.generate_normals()
		#Surf.aM,Surf.aN=0.5,0.5
		self.Surf.generate_collocations()



	def test_dWnvU_dzeta(self,PlotFlag=False):

		print('---------------------------------- Testing assembly.dWnvU_dzeta')

		Map,Surf=self.Map,self.Surf
		tsdata=self.tsdata
		ss=self.ss

		# generate non-zero field of external force
		u_ext0=tsdata.u_ext[ss]
		u_ext0[0,:,:]=u_ext0[0,:,:]-20.0
		u_ext0[1,:,:]=u_ext0[1,:,:]+60.0
		u_ext0[2,:,:]=u_ext0[2,:,:]+30.0
		u_ext0=u_ext0+np.random.rand(*u_ext0.shape)
		Surf.u_ext=u_ext0

		### analytical derivative
		Surf.get_input_velocities_at_collocation_points()
		Der=assembly.dWnvU_dzeta(Surf)

		### numerical derivative
		Surf.get_normal_input_velocities_at_collocation_points()
		u_norm0=Surf.u_input_coll_norm.copy()
		u_norm0_vec=u_norm0.reshape(-1,order='C')
		zeta0=Surf.zeta
		DerNum=np.zeros(Der.shape)

		Steps=np.array([1e-2,1e-3,1e-4,1e-5,1e-6])
		Er_max=0.0*Steps

		for ss in range(len(Steps)):
			step=Steps[ss]
			for jj in range(3*Map.Kzeta):
				# perturb
				cc_pert=Map.ind_3d_vert_vect[0][jj]
				mm_pert=Map.ind_3d_vert_vect[1][jj]
				nn_pert=Map.ind_3d_vert_vect[2][jj]
				zeta_pert=zeta0.copy()
				zeta_pert[cc_pert,mm_pert,nn_pert]+=step
				# calculate new normal velocity
				Surf_pert=surface.AeroGridSurface(Map,zeta=zeta_pert,u_ext=u_ext0,
													 gamma=tsdata.gamma[self.ss])
				Surf_pert.get_normal_input_velocities_at_collocation_points()
				u_norm_vec=Surf_pert.u_input_coll_norm.reshape(-1,order='C')
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


		




		

if __name__=='__main__':

	#unittest.main()
	T=Test_assembly()
	T.setUp()

	## Induced velocity
	T.test_dWnvU_dzeta()









