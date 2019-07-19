from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os
import sharpy.utils.settings as settings
import inspect
import shutil

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

    solver_classification = 'other'
    settings_types = dict()
    settings_description = dict()
    settings_default = dict()

    # Solver id for populating available_solvers[]
    @property
    def solver_id(self):
        raise NotImplementedError

    # The input is a ProblemData class structure
    @abstractmethod
    def initialise(self, data):
        pass

    # This executes the solver
    @abstractmethod
    def run(self):
        pass

    # @property
    def __doc__(self):
        # Generate documentation table
        settings_table = settings.SettingsTable()
        _doc = inspect.getdoc(self)
        _doc += settings_table.generate(settings_types, settings_default, settings_description)
        return _doc

def solver_from_string(string):
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

def dictionary_of_solvers():
    import sharpy.solvers
    import sharpy.postproc
    dictionary = dict()
    for solver in dict_of_solvers:
        if not solver.lower() == 'SaveData'.lower():
            # TODO: why it does not work for savedata?
            init_solver = initialise_solver(solver)
            dictionary[solver] = init_solver.settings_default

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
        route = sharpydir.SharpyDir + '/docs/source/includes/'
        if os.path.exists(route):
            shutil.rmtree(route)
    print('Creating documentation files for solvers in %s' %route)

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
            solver_folder = 'other'

        if solver_folder not in solver_types:
            solver_types.append(solver_folder)

        os.makedirs(route + '/' + solver_folder, exist_ok=True)
        title = k + '\n'
        title += len(k)*'-' + 2*'\n'
        if solver.__doc__ is not None:

            print('\tCreating %s' %(route + '/' + solver_folder + '/' + filename))
            autodoc_string = ''
            autodoc_string = '\n\n.. autoclass:: sharpy.solvers.' + k.lower() + '.'+ k + '\n\t:members:'
            with open(route + '/' + solver_folder + '/' + filename, "w") as out_file:
                out_file.write(title + autodoc_string)

    # Creates index files depending on the type of solver
    for solver_type in solver_types:
        filename = solver_type + '_solvers.rst'
        title = solver_type.capitalize() + ' Solvers'
        title += '\n' + len(title)*'+' + 2*'\n'
        with open(route + '/' + filename, "w") as out_file:
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


