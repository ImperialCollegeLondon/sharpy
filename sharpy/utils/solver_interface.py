from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os

# available_solvers = {}
dict_of_solvers = {}
# solver_types = {}
solvers = {}  # for internal working


# decorator
def solver(arg):
    # global available_solvers
    global dict_of_solvers
    try:
        arg.solver_id
    except AttributeError:
        raise AttributeError('Class defined as solver has no solver_id attribute')
    # try:
    #     arg.solver_type
    # except:
    #     raise AttributeError('Class defined as solver has no solver_type attribute')

    # try:
    #     available_solvers[arg.solver_type]
    # except KeyError:
    #     available_solvers[arg.solver_type] = []
    # available_solvers[arg.solver_type].append(arg.solver_id)
    dict_of_solvers[arg.solver_id] = arg
    # solver_types[arg.solver_id] = arg.solver_type
    return arg


def print_available_solvers():
    cout.cout_wrap('The available solvers on this session are:', 2)
    # for k, v in available_solvers.items():
    #     cout.cout_wrap('%s' % k, 2)
    #     for i in v:
    #         cout.cout_wrap('   %s' % i, 0)
    for name, i_solver in dict_of_solvers.items():
        cout.cout_wrap('%s ' % i_solver.solver_id, 2)


class BaseSolver(metaclass=ABCMeta):

    # Solver id for populating available_solvers[]
    @property
    def solver_id(self):
        raise NotImplementedError

    # @property
    # def solver_type(self):
    #     raise NotImplementedError

    # @property
    # def settings_types(self):
    #     raise NotImplementedError
    #
    # @property
    # def settings_default(self):
    #     raise NotImplementedError

    # The input is a ProblemData class structure
    @abstractmethod
    def initialise(self, data):
        pass

    # This executes the solver
    @abstractmethod
    def run(self):
        pass


def solver_from_string(string):
    # import sys
    # import functools
    return dict_of_solvers[string]


def solver_list_from_path(cwd):
    onlyfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

    for i_file in range(len(onlyfiles)):
        if ".py" in onlyfiles[i_file]:
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]
    return files


def initialise_solver(solver_name):
    cout.cout_wrap('Generating an instance of %s' % solver_name, 2)
    cls_type = solver_from_string(solver_name)
    solver = cls_type()
    return solver
