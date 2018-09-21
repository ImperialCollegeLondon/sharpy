
import copy
import numpy as np
import sys, os
import warnings
from IPython import embed
# sharpy
import sharpy.sharpy_main
import sharpy.utils.solver_interface as solver_interface
from sharpy.postproc.aeroforcescalculator import AeroForcesCalculator
import sharpy.utils.h5utils as h5



sys.path.append( os.environ["DIRuvlm3d_studies"]+'/templates' )
import geo_utils, flying_wings




# ------------------------------------------------------------------------------

# Define Parametrisation
N=12
M=4
Mstar_fact=10
Nsurf=1

# Flying properties
u_inf=25.
Nalpha=5
AlphaDeg=2.0

# sharpy setting
route_here=os.path.abspath('.') + '/test_savingh5/'
case_here='smith_Nsurf%.2dM%.2dN%.2dwk%.2d_a%.3d'\
							  %(Nsurf,M,N,Mstar_fact,int(np.round(10*AlphaDeg)))
### update parameters
os.system('mkdir -p %s'%(route_here,))

# Build wing model
ws=flying_wings.Smith(
                M=M,N=N,Mstar_fact=Mstar_fact,n_surfaces=2,
                u_inf=u_inf,alpha=AlphaDeg,
                route=route_here,
                case_name=case_here)
ws.clean_test_files()
ws.update_derived_params()
ws.generate_fem_file()
ws.generate_aero_file()
ws.set_default_config_dict()


# update default configuration
ws.config['SHARPy']['flow']=[
		'BeamLoader', 'AerogridLoader', 'StaticCoupled', 'SaveData']
ws.config['SaveData']={'folder': route_here,
					   'skip_attr': ['beam'],
					   }
ws.config.write()


### solve
data=sharpy.sharpy_main.main(['path_to_solver_useless',
	   							  			route_here+case_here+'.solver.txt'])

read=h5.readh5(route_here+case_here+'.data.h5')
