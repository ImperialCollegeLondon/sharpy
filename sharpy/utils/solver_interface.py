available_solvers = []

# decorator
def solver(arg):
    global available_solvers
    available_solvers.append(print(arg))

def print_available_solvers():
    print('The available solvers on this session are:\n')
    for solver in available_solvers:
        print('   %s\n'%(solver))
    print('\n')

from abc import ABCMeta, abstractmethod
class BaseSolver(metaclass=ABCMeta):

    # Solver id for populating available_solvers[]
    @property
    def solver_id(self):
        raise NotImplementedError

    # The input is a ProblemData class structure
    @abstractmethod
    def __init__(self, data):
        pass

    # This executes the solver
    @abstractmethod
    def run(self):
        pass
