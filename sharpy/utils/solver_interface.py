from abc import ABCMeta, abstractmethod
from sharpy.utils.cout_utils import cout_wrap
available_solvers = {}
dict_of_solvers = {}
solver_types = {}


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
    dict_of_solvers[arg.solver_id] = arg
    solver_types[arg.solver_id] = arg.solver_type
    return arg


def print_available_solvers():
    cout_wrap('The available solvers on this session are:')
    for k, v in available_solvers.items():
        cout_wrap('%s' % k)
        for i in v:
            cout_wrap('   %s' % i)


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


def solver_from_string(string):
    import sys
    import functools
    return dict_of_solvers[string]
    # return functools.reduce(getattr, string.split("."), sys.modules[__name__])
