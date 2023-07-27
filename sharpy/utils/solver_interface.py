from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os
import sharpy.utils.settings as settings
import inspect
import shutil
import sharpy.utils.exceptions as exceptions

dict_of_solvers = {}
solvers = {}  # for internal working


# decorator
def solver(arg):
    # global available_solvers
    global dict_of_solvers
    try:
        arg.solver_id
    except AttributeError:
        raise AttributeError('Class defined as solver has no solver_id attribute')
    dict_of_solvers[arg.solver_id] = arg

    # a = arg()
    # settings.SettingsTable().print(a)

    return arg


def print_available_solvers():
    cout.cout_wrap('The available solvers on this session are:', 2)
    for name, i_solver in dict_of_solvers.items():
        cout.cout_wrap('%s ' % i_solver.solver_id, 2)


class BaseSolver(metaclass=ABCMeta):

    # solver_classification = 'other'
    settings_types = dict()
    settings_description = dict()
    settings_default = dict()

    # Solver id for populating available_solvers[]
    @property
    def solver_id(self):
        raise NotImplementedError

    # The input is a ProblemData class structure
    @abstractmethod
    def initialise(self, data, restart=False):
        pass

    # This executes the solver
    @abstractmethod
    def run(self, **kwargs):
        pass

    # @property
    def __doc__(self):
        # Generate documentation table
        settings_table = settings.SettingsTable()
        _doc = inspect.getdoc(self)
        _doc += settings_table.generate(settings_types, settings_default, settings_description)
        return _doc

    def teardown(self):
        pass


def solver_from_string(string):
    try:
        solver = dict_of_solvers[string]
    except KeyError:
        raise exceptions.SolverNotFound(string)
    return solver


def solver_list_from_path(cwd):
    onlyfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

    for i_file in range(len(onlyfiles)):
        if onlyfiles[i_file].split('.')[-1] == 'py': # support autosaved files in the folder
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]
    return files


def initialise_solver(solver_name, print_info=True):
    if print_info:
        cout.cout_wrap('Generating an instance of %s' % solver_name, 2)
    cls_type = solver_from_string(solver_name)
    solver = cls_type()
    return solver


def dictionary_of_solvers(print_info=True):
    import sharpy.solvers
    import sharpy.postproc
    dictionary = dict()
    for solver in dict_of_solvers:
        if solver not in ['GridLoader', 'NonliftingBodyGridLoader']:
            init_solver = initialise_solver(solver, print_info)
            dictionary[solver] = init_solver.settings_default
        else:
            dictionary[solver] = {}

    return dictionary


def output_documentation(route=None):
    """
    Creates the ``.rst`` files for the solvers that have a docstring such that they can be parsed to Sphinx

    Args:
        route (str): Path to folder where solver files are to be created.

    """
    import sharpy.utils.sharpydir as sharpydir
    solver_types = []
    if route is None:
        base_route = sharpydir.SharpyDir + '/docs/source/includes/'
        route_solvers = base_route + 'solvers/'
        route_postprocs = base_route + 'postprocs/'
        if os.path.exists(route_solvers):
            print('Cleaning %s' % route_solvers)
            shutil.rmtree(route_solvers)
        if os.path.exists(route_postprocs):
            print('Cleaning %s', route_postprocs)
            shutil.rmtree(route_postprocs)

    print('Creating documentation files for solvers in %s' %route_solvers)
    print('Creating documentation files for post processors in %s' %route_postprocs)

    created_solvers = dict()

    for k, v in dict_of_solvers.items():
        if k[0] == '_':
            continue
        solver = v()
        created_solvers[k] = solver

        filename = k + '.rst'

        try:
            solver_folder = solver.solver_classification.lower()
        except AttributeError:
            print('The solver {} does not have a classification. Dumping it into "Other"'.format(k))
            solver_folder = 'other'
            if solver.__doc__ is None:
                continue

        if solver_folder == 'post-processor':
            solver_type = 'postprocessor'
            route_to_solver_python = 'sharpy.postproc.'
            folder = 'postprocs'
            solver_folder = '' # post-procs do not have sub classification unlike solvers
        else:
            solver_type = 'solver'
            route_to_solver_python = 'sharpy.solvers.'
            folder = 'solvers'
            if solver_folder not in solver_types:
                solver_types.append(solver_folder)

        if solver.solver_id == 'PreSharpy':
            route_to_solver_python = 'sharpy.presharpy.'

        os.makedirs(base_route + '/' + folder + '/' + solver_folder, exist_ok=True)
        title = k + '\n'
        title += len(k)*'-' + 2*'\n'
        if solver.__doc__ is not None:
            print('\tCreating %s' %(base_route + '/' + folder + '/' + solver_folder + '/' + filename))
            autodoc_string = ''
            autodoc_string = '\n\n.. autoclass:: ' + route_to_solver_python + k.lower() + '.' + k + '\n\t:members:'
            with open(base_route + '/' + folder + '/' + solver_folder + '/' + filename, "w") as out_file:
                out_file.write(title + autodoc_string)

    # Creates index files depending on the type of solver
    for solver_type in solver_types:
        if solver_type == 'post-processor':
            continue
        filename = solver_type + '_solvers.rst'
        title = solver_type.capitalize() + ' Solvers'
        title += '\n' + len(title)*'+' + 2*'\n'
        with open(route_solvers + '/' + filename, "w") as out_file:
            out_file.write(title)
            out_file.write('.. toctree::' + '\n')
            for k in dict_of_solvers.keys():
                if k[0] == '_':
                    continue
                try:
                    if created_solvers[k].solver_classification.lower() == solver_type and created_solvers[k].__doc__ is not None:
                        out_file.write('    ./' + solver_type + '/' + k + '\n')
                except AttributeError:
                    pass
