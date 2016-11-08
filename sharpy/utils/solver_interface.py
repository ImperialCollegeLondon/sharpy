from abc import ABCMeta, abstractmethod
available_solvers = {}


# decorator
def solver(arg):
    global available_solvers
    try:
        arg.solver_id
    except:
        raise AttributeError('Class defined as solver has no solver_id attribute')
    try:
        arg.solver_type
    except:
        raise AttributeError('Class defined as solver has no solver_type attribute')

    try:
        available_solvers[arg.solver_type]
    except KeyError:
        available_solvers[arg.solver_type] = []
    available_solvers[arg.solver_type].append(arg.solver_id)
    return arg


def print_available_solvers():
    print('The available solvers on this session are:')
    for k, v in available_solvers.items():
        print('%s' % k)
        for i in v:
            print('   %s' % i)


class BaseSolver(metaclass=ABCMeta):

    # Solver id for populating available_solvers[]
    @property
    def solver_id(self):
        raise NotImplementedError

    @property
    def solver_type(self):
        raise NotImplementedError

    # The input is a ProblemData class structure
    @abstractmethod
    def initialise(self, data):
        pass

    # This executes the solver
    @abstractmethod
    def run(self):
        pass
