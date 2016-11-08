import sys
import os

from presharpy.problemdata import ProblemData
from sharpy.beam.solver.nonlinearstatic import NonLinearStatic
import sharpy.utils.solver_interface as solver_interface

solver_interface.print_available_solvers()

if len(sys.argv) == 1:
    print('Running SHARPy using the default case name in the current directory:')
    print('./case.solver.txt is the main settings file')
    case_name = 'case'
    case_route = './'
elif len(sys.argv) == 2:
    case_name, case_route = os.path.split(sys.argv[1])
    print('Running SHARPy using the case: %s' % case_name)
    print('in the folder: %s' % case_route)
