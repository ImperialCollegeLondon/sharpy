import sys
import os

from presharpy.problemdata import ProblemData
from sharpy.beam.solver.nonlinearstatic import NonLinearStatic
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.input_arg as input_arg

solver_interface.print_available_solvers()

input_arg.read_settings()
