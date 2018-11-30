'''
Post-process aerodynamic force data
'''

import numpy as np
import save



def total_forces(data,Gframe=True):
	'''
	Compute total aerodynamic forces over all lifting surfaces.
	Requires 'AeroForcesCalculator' to be run.
	'''
	ts_max=len(data.structure.timestep_info)

	Fst=np.zeros((ts_max,6))
	Fun=np.zeros((ts_max,6))

	for tt in range(ts_max):
			
		if Gframe:
			faero_st=data.aero.timestep_info[tt].inertial_steady_forces
			faero_un=data.aero.timestep_info[tt].inertial_unsteady_forces
		else:
			faero_st=data.aero.timestep_info[tt].body_steady_forces
			faero_un=data.aero.timestep_info[tt].body_unsteady_forces

		# sum over surfaces
		Fst[tt,:]=np.sum(faero_st,axis=0)
		Fun[tt,:]=np.sum(faero_un,axis=0)

	return Fst,Fun



def saveh5(savedir,h5filename,data):
	'''
	Saves state of UVLM steady solution to h5 file.
	'''
	raise NameError('Function moved to save.save_aero!')













