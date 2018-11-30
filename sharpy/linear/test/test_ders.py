'''
Test elementary derivative methods
S. Maraniello, 4 Jun 2018
'''

import numpy as np 
import warnings
import unittest

from sharpy.utils.sharpydir import SharpyDir
import sharpy.utils.ctypes_utils as ct_utils

import sharpy.linear.src.lib_dbiot as dbiot 
import sharpy.linear.src.libuvlm as libuvlm 
# from IPython import embed


class Test_ders(unittest.TestCase):
	'''
	Test methods into assembly module
	'''

	def setUp(self):

		self.zetaP=np.array([3.0,5.5,2.0])
		self.zeta0=np.array([1.0,3.0,0.9])
		self.zeta1=np.array([5.0,3.1,1.9])
		self.zeta2=np.array([4.8,8.1,2.5])
		self.zeta3=np.array([0.9,7.9,1.7])




	def test_dbiot_segment(self):
		print('\n-------------------------------------- Testing dbiot.eval_seg')

		gamma=2.4
		zetaP=self.zetaP
		zetaA=self.zeta1
		zetaB=self.zeta2
		Q0=libuvlm.biot_segment(zetaP,zetaA,zetaB,gamma)

		### compare different analytical derivative
		DerP_an,DerA_an,DerB_an=dbiot.eval_seg_exp(zetaP,zetaA,zetaB,gamma)
		DerP_an2,DerA_an2,DerB_an2=dbiot.eval_seg_comp(zetaP,zetaA,zetaB,gamma)
		er_max=max( np.max(np.abs(DerP_an2-DerP_an)),
							np.max(np.abs(DerA_an2-DerA_an)),
								np.max(np.abs(DerB_an2-DerB_an)) )
		assert er_max<1e-16, 'Analytical models not matching' 


		### compare vs numerical derivative
		Steps=np.array([1e-2,1e-4,1e-6])
		Er_max=0.0*Steps
		for ss in range(len(Steps)):
			step=Steps[ss]
			DerP_num=0.0*DerP_an
			DerA_num=0.0*DerA_an
			DerB_num=0.0*DerB_an
			for cc_zeta in range(3):
				dzeta=np.zeros((3,))
				dzeta[cc_zeta]=step
				DerP_num[:,cc_zeta]=(
					libuvlm.biot_segment(zetaP+dzeta,zetaA,zetaB,gamma)-Q0)/step
				DerA_num[:,cc_zeta]=(
					libuvlm.biot_segment(zetaP,zetaA+dzeta,zetaB,gamma)-Q0)/step
				DerB_num[:,cc_zeta]=(
					libuvlm.biot_segment(zetaP,zetaA,zetaB+dzeta,gamma)-Q0)/step
			er_max=max( np.max(np.abs(DerP_num-DerP_an)),
							np.max(np.abs(DerA_num-DerA_an)),
								np.max(np.abs(DerB_num-DerB_an)) )
			print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
			assert er_max<5e1*step, 'Error larger than 50 times step size'
			Er_max[ss]=er_max	




	def test_dbiot_segment_mid(self):
		print('\n------------------------- Testing dbiot.eval_seg at mid-point')

		gamma=2.4
		zetaA=self.zeta1
		zetaB=self.zeta2
		zetaP=.3*zetaA+0.7*zetaB

		Q0=libuvlm.biot_segment(zetaP,zetaA,zetaB,gamma)

		### compare different analytical derivative
		DerP_an,DerA_an,DerB_an=dbiot.eval_seg_exp(zetaP,zetaA,zetaB,gamma)
		DerP_an2,DerA_an2,DerB_an2=dbiot.eval_seg_comp(zetaP,zetaA,zetaB,gamma)
		er_max=max( np.max(np.abs(DerP_an2-DerP_an)),
							np.max(np.abs(DerA_an2-DerA_an)),
								np.max(np.abs(DerB_an2-DerB_an)) )
		assert er_max<1e-16, 'Analytical models not matching' 


		### compare vs numerical derivative
		Steps=np.array([1e-2,1e-4,1e-6])
		Er_max=0.0*Steps
		for ss in range(len(Steps)):
			step=Steps[ss]
			DerP_num=0.0*DerP_an
			DerA_num=0.0*DerA_an
			DerB_num=0.0*DerB_an
			for cc_zeta in range(3):
				dzeta=np.zeros((3,))
				dzeta[cc_zeta]=step
				DerP_num[:,cc_zeta]=(
					libuvlm.biot_segment(zetaP+dzeta,zetaA,zetaB,gamma)-Q0)/step
				DerA_num[:,cc_zeta]=(
					libuvlm.biot_segment(zetaP,zetaA+dzeta,zetaB,gamma)-Q0)/step
				DerB_num[:,cc_zeta]=(
					libuvlm.biot_segment(zetaP,zetaA,zetaB+dzeta,gamma)-Q0)/step
			er_max=max( np.max(np.abs(DerP_num-DerP_an)),
							np.max(np.abs(DerA_num-DerA_an)),
								np.max(np.abs(DerB_num-DerB_an)) )
			print('FD step: %.2e ---> Max error: %.2e'%(step,er_max) )
			assert er_max<5e1*step, 'Error larger than 50 times step size'
			Er_max[ss]=er_max	



	def test_dbiot_panel(self):
		print('\n---------------------------------- Testing dbiot.eval_panel_*')

		gamma=2.4
		zetaP=self.zetaP
		zeta0=self.zeta0
		zeta1=self.zeta1
		zeta2=self.zeta2
		zeta3=self.zeta3

		ZetaPanel=np.array([zeta0,zeta1,zeta2,zeta3])
		Q0=libuvlm.biot_panel(zetaP,ZetaPanel,gamma)

		# compare analytical derivatives models
		DerP_an,DerVer_an=dbiot.eval_panel_exp(zetaP,ZetaPanel,gamma)
		DerP_an2,DerVer_an2=dbiot.eval_panel_comp(zetaP,ZetaPanel,gamma)
		DerP_an3,DerVer_an3=dbiot.eval_panel_fast(zetaP,ZetaPanel,gamma)
		DerP_an4,DerVer_an4=dbiot.eval_panel_cpp(zetaP,ZetaPanel,gamma)

		er_max=max( np.max(np.abs(DerP_an2-DerP_an)),
										   np.max(np.abs(DerVer_an2-DerVer_an)))
		assert er_max<1e-16, 'eval_panel_comp not matching with eval_panel_exp' 
		er_max=max( np.max(np.abs(DerP_an3-DerP_an)),
										   np.max(np.abs(DerVer_an3-DerVer_an)))
		assert er_max<1e-16, 'eval_panel_fast not matching with eval_panel_exp' 
		er_max=max( np.max(np.abs(DerP_an4-DerP_an)),
										   np.max(np.abs(DerVer_an4-DerVer_an)))
		assert er_max<1e-16, 'eval_panel_cpp not matching with eval_panel_exp' 


		# compare vs. numerical derivative
		Steps=np.array([1e-2,1e-4,1e-6])
		ErP_max=0.0*Steps
		ErVer_max=0.0*Steps
		for ss in range(len(Steps)):
			step=Steps[ss]
			DerP_num=0.0*DerP_an
			DerVer_num=0.0*DerVer_an

			### Perturb component
			for cc in range(3):
				dzeta=np.zeros((3,))
				dzeta[cc]=step

				# derivative w.r.t. target point
				DerP_num[:,cc]=\
					   (libuvlm.biot_panel(zetaP+dzeta,ZetaPanel,gamma)-Q0)/step

				# derivative w.r.t panel vertices
				for vv in range(4):
					ZetaPanel_pert=ZetaPanel.copy()
					ZetaPanel_pert[vv,:]+=dzeta
					DerVer_num[vv,:,cc]=\
						(libuvlm.biot_panel(zetaP,ZetaPanel_pert,gamma)-Q0)/step

			erP_max=np.max(np.abs(DerP_num-DerP_an))
			erVer_max=np.max(np.abs(DerVer_num-DerVer_an))
			print('FD step: %.2e ---> Max error (P,Vert): (%.2e,%.2e)'\
													  %(step,erP_max,erVer_max))
			assert erP_max<5e1*step,\
					 	     'Error w.r.t. zetaP larger than 50 times step size'
			assert erVer_max<5e1*step,\
					 	 'Error w.r.t. ZetaPanel larger than 50 times step size'
			ErP_max[ss]=erP_max	
			ErVer_max[ss]=erVer_max	

		# assert monothony
		for ss in range(len(Steps)-1):
			assert ErP_max[ss+1]<ErP_max[ss],\
				'Error of derivative w.r.t. zetaP not decreasing monothonically'
			assert ErVer_max[ss+1]<ErVer_max[ss],\
			'Error of derivative w.r.t. ZetaPanel not decreasing monothonically'




	def test_dbiot_panel_mid_segment(self):
		print('\n-------------- Testing dbiot.eval_panel with zetaP on segment')

		gamma=2.4
		zeta0=self.zeta0
		zeta1=self.zeta1
		zeta2=self.zeta2
		zeta3=self.zeta3
		zetaP=0.3*zeta1+0.7*zeta2

		ZetaPanel=np.array([zeta0,zeta1,zeta2,zeta3])
		Q0=libuvlm.biot_panel(zetaP,ZetaPanel,gamma)

		# compare analytical derivatives models
		DerP_an,DerVer_an=dbiot.eval_panel_exp(zetaP,ZetaPanel,gamma)
		DerP_an2,DerVer_an2=dbiot.eval_panel_comp(zetaP,ZetaPanel,gamma)
		DerP_an3,DerVer_an3=dbiot.eval_panel_fast(zetaP,ZetaPanel,gamma)
		DerP_an4,DerVer_an4=dbiot.eval_panel_cpp(zetaP,ZetaPanel,gamma)

		er_max=max( np.max(np.abs(DerP_an2-DerP_an)),
										   np.max(np.abs(DerVer_an2-DerVer_an)))
		assert er_max<1e-16, 'eval_panel_comp not matching with eval_panel_exp' 
		er_max=max( np.max(np.abs(DerP_an3-DerP_an)),
										   np.max(np.abs(DerVer_an3-DerVer_an)))
		assert er_max<1e-16, 'eval_panel_fast not matching with eval_panel_exp' 
		er_max=max( np.max(np.abs(DerP_an4-DerP_an)),
										   np.max(np.abs(DerVer_an4-DerVer_an)))
		assert er_max<1e-16, 'eval_panel_cpp not matching with eval_panel_exp' 


		# compare vs. numerical derivative
		Steps=np.array([1e-2,1e-4,1e-6])
		ErP_max=0.0*Steps
		ErVer_max=0.0*Steps
		for ss in range(len(Steps)):
			step=Steps[ss]
			DerP_num=0.0*DerP_an
			DerVer_num=0.0*DerVer_an

			### Perturb component
			for cc in range(3):
				dzeta=np.zeros((3,))
				dzeta[cc]=step

				# derivative w.r.t. target point
				DerP_num[:,cc]=\
					   (libuvlm.biot_panel(zetaP+dzeta,ZetaPanel,gamma)-Q0)/step

				# derivative w.r.t panel vertices
				for vv in range(4):
					ZetaPanel_pert=ZetaPanel.copy()
					ZetaPanel_pert[vv,:]+=dzeta
					DerVer_num[vv,:,cc]=\
						(libuvlm.biot_panel(zetaP,ZetaPanel_pert,gamma)-Q0)/step

			erP_max=np.max(np.abs(DerP_num-DerP_an))
			erVer_max=np.max(np.abs(DerVer_num-DerVer_an))
			print('FD step: %.2e ---> Max error (P,Vert): (%.2e,%.2e)'\
													  %(step,erP_max,erVer_max))
			assert erP_max<5e1*step,\
					 	     'Error w.r.t. zetaP larger than 50 times step size'
			assert erVer_max<5e1*step,\
					 	 'Error w.r.t. ZetaPanel larger than 50 times step size'
			ErP_max[ss]=erP_max	
			ErVer_max[ss]=erVer_max	

		# assert monothony
		for ss in range(len(Steps)-1):
			assert ErP_max[ss+1]<ErP_max[ss],\
				'Error of derivative w.r.t. zetaP not decreasing monothonically'
			assert ErVer_max[ss+1]<ErVer_max[ss],\
			'Error of derivative w.r.t. ZetaPanel not decreasing monothonically'


		

if __name__=='__main__':

	# unittest.main()

	T=Test_ders()
	T.setUp()
	T.test_dbiot_segment()
	T.test_dbiot_segment_mid()
	T.test_dbiot_panel()
	T.test_dbiot_panel_mid_segment()











