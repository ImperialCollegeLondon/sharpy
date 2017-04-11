import os
import time
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.input_arg as input_arg
import sharpy.utils.cout_utils as cout

# solver list -- It is important to import them here
from presharpy.problemdata import ProblemData
from sharpy.beam.solver.nonlinearstatic import NonLinearStatic
from sharpy.beam.solver.nonlineardynamic import NonLinearDynamic
from sharpy.beam.solver.vibrationmodes import VibrationModes
from sharpy.beam.postproc.vibrationmodesplot import VibrationModesPlot
from sharpy.beam.postproc.staticplot import StaticPlot

from sharpy.aero.solver.staticuvlm import StaticUvlm
# ------------

# timing
t = time.process_time()
# Hi! message
cout.cout_wrap(cout.sharpy_ascii)
cout.cout_wrap(cout.sharpy_license)
cwd = os.getcwd()
cout.cout_wrap('Running SHARPy from ' + cwd)

solver_interface.print_available_solvers()
settings = input_arg.read_settings()

# Loop for the solvers specified in *.solver.txt['SHARPy']['flow']
# run preSHARPy
data = ProblemData(settings)
for solver_name in settings['SHARPy']['flow']:
    print('Generating an instance of %s' % solver_name)
    cls_type = solver_interface.solver_from_string(solver_name)
    solver = cls_type()
    solver.initialise(data)
    data = solver.run()

elapsed_time = time.process_time() - t
print('FINISHED - Elapsed time = ' + str(elapsed_time) + ' seconds')
