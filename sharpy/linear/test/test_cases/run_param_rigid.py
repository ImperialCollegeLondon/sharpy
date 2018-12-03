'''
Parametric study with angle of attack.
'''

import copy
import numpy as np
import sys, os
import warnings

import sharpy.sharpy_main
import sharpy.utils.solver_interface as solver_interface
from sharpy.postproc.aeroforcescalculator import AeroForcesCalculator
import sharpy.utils.h5utils as h5
import sharpy.utils.algebra as algebra
import cases.templates.flying_wings as flying_wings


# ------------------------------------------------------------------------------

# Define Parametrisation
M,N,Mstar_fact= 20,40,20
# M,N,Mstar_fact= 12,40,16
M,N,Mstar_fact= 4,40,20
Nsurf=1
u_inf=150.


# Flying properties
#pvec=np.linspace(-1,1,61)
pvec=np.linspace(-1,1,31)

AlphaFoRA=0.
SideMidDeg=0.
SideMaxDeg=6.
AlphaInfMid=0.
AlphaInfMax=3.
AlphaInfVecDeg=AlphaInfMid+AlphaInfMax*pvec
AlphaVecDeg=0.*pvec+AlphaFoRA
SideVecDeg=SideMidDeg+SideMaxDeg*pvec
RollVecDeg=0.*pvec
Npoints=len(AlphaVecDeg)

# sharpy setting
route_main=os.path.abspath('.') + '/cases/rigid/'
case_main='goland_mod_rig_Nsurf%.2dM%.2dN%.2dwk%.2d' %(Nsurf,M,N,Mstar_fact)



for ii in range(Npoints):

	### update parameters
	tplparams=(int(np.round(100*AlphaInfVecDeg[ii])),
				int(np.round(100*AlphaVecDeg[ii])),
			   int(np.round(100*SideVecDeg[ii])) ,
			   int(np.round(100*RollVecDeg[ii])) )
	case_here=case_main+'_ainf%.4da%.4ds%.4dr%.4d'%tplparams 
	route_here=route_main
	os.system('mkdir -p %s'%(route_here,))


	### ------------------------------------------------------ Build wing model
	ws=flying_wings.Goland( M=M,N=N,Mstar_fact=Mstar_fact,n_surfaces=Nsurf,
								u_inf=u_inf,
								alpha=AlphaVecDeg[ii],
								beta=-SideVecDeg[ii],
								route=route_here,
								case_name=case_here)
	# updte wind direction
	quat_wind=algebra.euler2quat(-np.pi/180.*np.array([0.,AlphaInfVecDeg[ii],0.]))
	u_inf_dir=np.dot( algebra.quat2rotation(quat_wind),np.array([1.,0.,0.]))

	ws.main_ea-=.25/M
	ws.main_cg-=.25/M

	ws.root_airfoil_P = 4
	ws.root_airfoil_M = 2
	ws.tip_airfoil_P = 4
	ws.tip_airfoil_M = 2

	ws.clean_test_files()
	ws.update_derived_params()
	ws.generate_fem_file()
	ws.generate_aero_file()

	### solution flow
	ws.set_default_config_dict()
	ws.config['SHARPy']['flow']=[
			'BeamLoader','AerogridLoader','StaticUvlm','Modal',
													  'AerogridPlot','SaveData']
	ws.config['SaveData']={'folder': route_here}
	str_uinf_dir=[ '%.6f'%u_inf_dir[cc] for cc in [0,1,2]]
	ws.config['AerogridLoader']['freestream_dir']=str_uinf_dir
	ws.config['StaticUvlm']['velocity_field_input']['u_inf_direction']=u_inf_dir
	ws.config['StaticCoupled']['aero_solver_settings']['velocity_field_input']['u_inf_direction']=u_inf_dir
	if M<5: 
		ws.config['StaticCoupled']['tolerance']=1e-7
	ws.config.write()


	### solve
	data=sharpy.sharpy_main.main(['path_to_solver_useless',
	   							  			route_here+case_here+'.solver.txt'])





#embed()

