import os
import time

import sharpy.utils.cout_utils as cout
import sharpy.utils.input_arg as input_arg
import sharpy.utils.sharpydir as sharpydir
import sharpy.utils.solver_interface as solver_interface
from sharpy.presharpy.presharpy import PreSharpy

from sharpy.presharpy.presharpy import PreSharpy
# Loading solvers and postprocessors
import sharpy.solvers
import sharpy.postproc
# ------------

# timing
t = time.process_time()
# Hi! message
cout.cout_wrap(cout.sharpy_ascii)
cout.cout_wrap(cout.sharpy_license)
cwd = os.getcwd()
cout.cout_wrap('Running SHARPy from ' + cwd, 2)
cout.cout_wrap('SHARPy version being run is in ' + sharpydir.SharpyDir, 2)

solver_interface.print_available_solvers()
settings = input_arg.read_settings()

# Loop for the solvers specified in *.solver.txt['SHARPy']['flow']
# run preSHARPy
data = PreSharpy(settings)
for solver_name in settings['SHARPy']['flow']:
    solver = solver_interface.initialise_solver(solver_name)
    solver.initialise(data)
    data = solver.run()

elapsed_time = time.process_time() - t
cout.cout_wrap('FINISHED - Elapsed time = ' + str(elapsed_time) + ' seconds', 2)
